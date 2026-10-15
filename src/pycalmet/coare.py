"""Vectorized CALMET 6.5 COARE 2.6 overwater surface fluxes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


MODEL_DTYPE=np.float32
ACCUM_DTYPE=np.float64


@dataclass(frozen=True, slots=True)
class CoareFlux:
    """Fields returned by CALMET's ``COARE -> BULK_FLUX -> ASL`` chain.

    Attributes:
        roughness_m: Aerodynamic roughness length in metres.
        monin_obukhov_m: Monin-Obukhov length in metres.
        friction_velocity_m_s: Friction velocity in metres per second.
        temperature_scale_k: Turbulent temperature scale in kelvin.
        humidity_scale_kg_kg: Turbulent specific-humidity scale in kilograms
            per kilogram.
        sensible_heat_flux_w_m2: Upward sensible heat flux in watts per square
            metre.
        latent_heat_flux_w_m2: Upward latent heat flux in watts per square
            metre.
        buoyancy_flux_k_m_s: Kinematic virtual-temperature flux in
            kelvin-metres per second.
        cool_skin_delta_k: Cool-skin temperature difference in kelvin.
        gustiness_m_s: Convective gustiness in metres per second.
        stress_n_m2: Along-wind surface stress in newtons per square metre.
        rain_heat_flux_w_m2: Rain heat flux in watts per square metre.

    Notes:
        Every array has the broadcast shape of the inputs passed to
        :func:`surface_flux` and uses the model's ``float32`` dtype.
    """

    roughness_m: np.ndarray
    monin_obukhov_m: np.ndarray
    friction_velocity_m_s: np.ndarray
    temperature_scale_k: np.ndarray
    humidity_scale_kg_kg: np.ndarray
    sensible_heat_flux_w_m2: np.ndarray
    latent_heat_flux_w_m2: np.ndarray
    buoyancy_flux_k_m_s: np.ndarray
    cool_skin_delta_k: np.ndarray
    gustiness_m_s: np.ndarray
    stress_n_m2: np.ndarray
    rain_heat_flux_w_m2: np.ndarray


@dataclass(frozen=True, slots=True)
class WarmLayerState:
    """Per-grid state carried by CALMET's COARE ``IWARM=1`` option.

    Attributes:
        time_old_s: Previous local solar time in seconds after midnight.
        heat_column_j_m2: Accumulated warm-layer heat in joules per square
            metre.
        stress_integral_n_s_m2: Time-integrated surface stress in
            newton-seconds per square metre.
        stress_old_n_m2: Surface stress from the previous step.
        rain_heat_old_w_m2: Previous rain heat flux.
        sensible_old_w_m2: Previous sensible heat flux.
        latent_old_w_m2: Previous latent heat flux.
        morning_started: Whether daytime heat accumulation has begun.
        absorption_fraction: Fraction of shortwave radiation absorbed in the
            warm layer.
        layer_thickness_m: Warm-layer thickness in metres.
        initialized: Whether a grid cell has a previous timestep.

    Notes:
        Accumulators use ``float64`` to limit drift across long simulations;
        ordinary CALMET fields remain ``float32``.
    """

    time_old_s: np.ndarray
    heat_column_j_m2: np.ndarray
    stress_integral_n_s_m2: np.ndarray
    stress_old_n_m2: np.ndarray
    rain_heat_old_w_m2: np.ndarray
    sensible_old_w_m2: np.ndarray
    latent_old_w_m2: np.ndarray
    morning_started: np.ndarray
    absorption_fraction: np.ndarray
    layer_thickness_m: np.ndarray
    initialized: np.ndarray


def initial_warm_layer_state(shape):
    """Create COARE first-call state for an output-grid shape.

    Args:
        shape: Integer or tuple defining the horizontal array shape.

    Returns:
        Independent warm-layer state arrays initialized to the COARE defaults.
    """
    # Warm-layer heat and stress integrals persist across many timesteps, so
    # keep this small state in FP64 while ordinary model fields use FP32.
    zeros=np.zeros(shape,dtype=ACCUM_DTYPE)
    return WarmLayerState(
        zeros.copy(),zeros.copy(),zeros.copy(),zeros.copy(),zeros.copy(),
        zeros.copy(),zeros.copy(),np.zeros(shape,dtype=bool),
        np.full(shape,.5),np.full(shape,19.),np.zeros(shape,dtype=bool))


def advance_warm_layer(state,solar_time_s,sea_temperature_k,sensor_depth_m,
                       shortwave_down_w_m2,longwave_down_w_m2):
    """Advance the vectorized COARE PWP warm layer before the ASL solve.

    Args:
        state: State retained from the preceding timestep.
        solar_time_s: Local solar time in seconds after midnight.
        sea_temperature_k: Measured sea temperature in kelvin.
        sensor_depth_m: Sea-temperature sensor depth in metres.
        shortwave_down_w_m2: Downward shortwave radiation in watts per square
            metre.
        longwave_down_w_m2: Downward longwave radiation in watts per square
            metre.

    Returns:
        A three-item tuple containing adjusted sea temperature in kelvin,
        updated warm-layer state, and the bulk warm-layer increment in kelvin.
        Numeric arrays have the broadcast input shape; temperatures use
        ``float32``.
    """
    time,temperature,depth,shortwave,longwave=np.broadcast_arrays(
        np.asarray(solar_time_s,dtype=ACCUM_DTYPE),
        np.asarray(sea_temperature_k,dtype=ACCUM_DTYPE),
        np.asarray(sensor_depth_m,dtype=ACCUM_DTYPE),
        np.asarray(shortwave_down_w_m2,dtype=ACCUM_DTYPE),
        np.asarray(longwave_down_w_m2,dtype=ACCUM_DTYPE))
    initialized=np.asarray(state.initialized,dtype=bool)
    midnight=initialized&(time<state.time_old_s)
    normal=initialized&~midnight
    heat=np.where(midnight,0.,state.heat_column_j_m2).copy()
    stress_integral=np.where(midnight,0.,state.stress_integral_n_s_m2).copy()
    started=np.where(midnight,False,state.morning_started).copy()
    fraction=np.where(midnight,.5,state.absorption_fraction).copy()
    thickness=np.where(midnight,19.,state.layer_thickness_m).copy()
    rns=.945*shortwave
    rnl=.97*(5.67e-8*(temperature-273.15+273.16)**4-longwave)
    heat_out=rnl+state.sensible_old_w_m2+state.latent_old_w_m2+state.rain_heat_old_w_m2
    effective=fraction*rns-heat_out
    integrate=normal&((effective>=50.)|started)
    started|=integrate
    elapsed=time-state.time_old_s
    stress_integral=np.where(
        integrate,stress_integral+np.maximum(state.stress_old_n_m2,.002)*elapsed,
        stress_integral)
    expansion=2.1e-5*np.power(temperature-273.15+3.2,.79)
    ctd1=np.sqrt(2.*.65*4000./(expansion*9.81*1022.))
    for _ in range(5):
        fraction_candidate=(1.-(.28*.014*(1.-np.exp(-thickness/.014))+
                                  .27*.357*(1.-np.exp(-thickness/.357))+
                                  .45*12.82*(1.-np.exp(-thickness/12.82)))/thickness)
        joule=(fraction_candidate*rns-heat_out)*elapsed
        positive=heat+joule>0.
        thickness_candidate=np.minimum(
            19.,ctd1*stress_integral/np.sqrt(np.maximum(
                heat+joule,np.finfo(ACCUM_DTYPE).tiny)))
        fraction=np.where(integrate,fraction_candidate,fraction)
        thickness=np.where(integrate&positive,thickness_candidate,thickness)
    joule=(fraction*rns-heat_out)*elapsed
    nonpositive=integrate&((heat+joule)<=0.)
    fraction=np.where(nonpositive,.76,fraction)
    thickness=np.where(nonpositive,19.,thickness)
    joule=np.where(nonpositive,(fraction*rns-heat_out)*elapsed,joule)
    heat=np.where(integrate,heat+joule,heat)
    ctd2=np.sqrt(2.*expansion*9.81/(.65*1022.))/(4000.**1.5)
    warming=np.where(integrate&(heat>0.),ctd2*np.maximum(heat,0.)**1.5/
                     np.maximum(stress_integral,np.finfo(ACCUM_DTYPE).tiny),0.)
    sensor_increment=np.where(thickness<depth,warming,warming*depth/thickness)
    adjusted=np.where(normal,temperature+sensor_increment,temperature)
    next_state=WarmLayerState(
        time,heat,stress_integral,state.stress_old_n_m2,state.rain_heat_old_w_m2,
        state.sensible_old_w_m2,state.latent_old_w_m2,started,fraction,thickness,
        np.ones(time.shape,dtype=bool))
    return (np.asarray(adjusted,dtype=MODEL_DTYPE),next_state,
            np.asarray(warming,dtype=MODEL_DTYPE))


def update_warm_layer_fluxes(state,flux):
    """Store surface fluxes consumed by the next warm-layer timestep.

    Args:
        state: Warm-layer state after :func:`advance_warm_layer`.
        flux: COARE flux fields calculated for the current timestep.

    Returns:
        A new state whose previous-flux arrays use ``float64`` while all other
        state arrays are retained.
    """
    return WarmLayerState(
        state.time_old_s,state.heat_column_j_m2,state.stress_integral_n_s_m2,
        np.asarray(flux.stress_n_m2,dtype=ACCUM_DTYPE),
        np.asarray(flux.rain_heat_flux_w_m2,dtype=ACCUM_DTYPE),
        np.asarray(flux.sensible_heat_flux_w_m2,dtype=ACCUM_DTYPE),
        np.asarray(flux.latent_heat_flux_w_m2,dtype=ACCUM_DTYPE),
        state.morning_started,state.absorption_fraction,state.layer_thickness_m,
        state.initialized)


def distance_to_coast(landuse,water_range,dx_km,dy_km):
    """Calculate CALMET SETCOAST distance to the nearest land-cell centre.

    Args:
        landuse: Two-dimensional integer array of dominant land-use classes.
        water_range: Inclusive ``(minimum, maximum)`` water-category codes.
        dx_km: East-west grid spacing in kilometres.
        dy_km: North-south grid spacing in kilometres.

    Returns:
        ``float32`` array in kilometres with the land-use shape. Land cells
        are zero; an all-water domain receives the CALMET sentinel ``1e6``.
    """
    categories=np.asarray(landuse)
    water=(categories>=water_range[0])&(categories<=water_range[1])
    if water.all():
        return np.where(water,MODEL_DTYPE(1.e6),MODEL_DTYPE(0.)).astype(MODEL_DTYPE)
    distance=distance_transform_edt(water,sampling=(dy_km,dx_km))
    return np.where(water,distance,0.).astype(MODEL_DTYPE)


def _humidity_vapor_pressure(temperature_c, pressure_mb):
    """Evaluate CALMET's Buck/Tetens saturation vapor pressure.

    Args:
        temperature_c: Scalar or array air temperature in degrees Celsius.
        pressure_mb: Broadcast-compatible pressure in millibars.

    Returns:
        Saturation vapor pressure in millibars with ``float32`` arithmetic.
    """
    temperature=np.asarray(temperature_c,dtype=MODEL_DTYPE)
    pressure=np.asarray(pressure_mb,dtype=MODEL_DTYPE)
    return ((1.0007+3.46e-6*pressure)*6.1121*
            np.exp(17.502*temperature/(240.97+temperature)))


def _psiu(z_over_l):
    """Evaluate the COARE momentum stability correction.

    Args:
        z_over_l: Scalar or array dimensionless stability coordinate ``z/L``.

    Returns:
        ``float32`` correction array with the input shape.
    """
    value=np.asarray(z_over_l,dtype=MODEL_DTYPE)
    unstable=value<0.
    result=np.empty_like(value)
    if unstable.any():
        selected=value[unstable]
        x=np.power(np.maximum(1.-15.*selected,0.),.25)
        kansas=(2.*np.log((1.+x)/2.)+np.log((1.+x*x)/2.)-
                2.*np.arctan(x)+np.pi/2.)
        y=np.cbrt(np.maximum(1.-10.15*selected,0.))
        convective=(1.5*np.log((1.+y+y*y)/3.)-
                    np.sqrt(3.)*np.arctan((1.+2.*y)/np.sqrt(3.))+
                    np.pi/np.sqrt(3.))
        weight=selected*selected/(1.+selected*selected)
        result[unstable]=(1.-weight)*kansas+weight*convective
    stable=~unstable
    if stable.any():
        selected=value[stable]
        result[stable]=-((1.+selected)+(2./3.)*(selected-14.28)/
                         np.exp(np.minimum(50.,.35*selected))+8.525)
    return result


def _psit(z_over_l):
    """Evaluate the COARE scalar-temperature stability correction.

    Args:
        z_over_l: Scalar or array dimensionless stability coordinate ``z/L``.

    Returns:
        ``float32`` correction array with the input shape.
    """
    value=np.asarray(z_over_l,dtype=MODEL_DTYPE)
    unstable=value<0.
    result=np.empty_like(value)
    if unstable.any():
        selected=value[unstable]
        x=np.sqrt(np.maximum(1.-15.*selected,0.))
        kansas=2.*np.log((1.+x)/2.)
        y=np.cbrt(np.maximum(1.-34.15*selected,0.))
        convective=(1.5*np.log((1.+y+y*y)/3.)-
                    np.sqrt(3.)*np.arctan((1.+2.*y)/np.sqrt(3.))+
                    np.pi/np.sqrt(3.))
        weight=selected*selected/(1.+selected*selected)
        result[unstable]=(1.-weight)*kansas+weight*convective
    stable=~unstable
    if stable.any():
        selected=value[stable]
        result[stable]=-(np.maximum(1.+2.*selected/3.,0.)**1.5+
                         (2./3.)*(selected-14.28)/
                         np.exp(np.minimum(50.,.35*selected))+8.525)
    return result


def downward_longwave(temperature_k,relative_humidity_percent,cloud_fraction,
                      ceiling_height_m):
    """Calculate downward longwave radiation with CALMET ``RADFLX``.

    Args:
        temperature_k: Air temperature in kelvin.
        relative_humidity_percent: Relative humidity in percent.
        cloud_fraction: Cloud cover fraction from zero to one.
        ceiling_height_m: Cloud-ceiling height in metres.

    Returns:
        Broadcast ``float32`` array of downward longwave radiation in watts
        per square metre.
    """
    temperature,humidity,cloud,ceiling=np.broadcast_arrays(
        np.asarray(temperature_k,dtype=MODEL_DTYPE),
        np.asarray(relative_humidity_percent,dtype=MODEL_DTYPE),
        np.asarray(cloud_fraction,dtype=MODEL_DTYPE),
        np.asarray(ceiling_height_m,dtype=MODEL_DTYPE))
    water=np.clip(.439*(humidity*.01)*np.exp(26.23-5416./temperature)/temperature,.1,5.)
    clear=(.725+.17*np.log10(water))*5.67e-8*temperature**4
    hundreds_ft=ceiling/30.48
    enhancement=np.where(hundreds_ft<64.,.26,
                         np.where(hundreds_ft<=210.,.22,.006))*cloud
    return np.asarray(clear*(1.+enhancement),dtype=MODEL_DTYPE)


def surface_flux(
    wind_speed_m_s,
    air_temperature_k,
    sea_temperature_k,
    humidity,
    *,
    wind_height_m=10.,
    temperature_height_m=10.,
    pressure_mb=1008.,
    boundary_layer_height_m=600.,
    shortwave_down_w_m2=0.,
    longwave_down_w_m2=0.,
    rainfall_mm_h=0.,
    wave_method=0,
    wave_period_s=-999.,
    wave_height_m=-999.,
    coastal_shelf_length_km=0.,
    distance_to_coast_km=0.,
    cool_skin=False,
):
    """Evaluate the stateless part of CALMET's COARE 2.6 implementation.

    ``humidity`` follows the legacy interface: values below 2 are relative
    humidity fractions, while larger values are specific humidity in g/kg.
    Warm-layer storage is intentionally separate because it carries state
    between timesteps; this function exactly covers ``IWARM=0``.

    Args:
        wind_speed_m_s: Wind speed at ``wind_height_m`` in metres per second.
        air_temperature_k: Air temperature at ``temperature_height_m`` in
            kelvin.
        sea_temperature_k: Bulk sea-surface temperature in kelvin.
        humidity: Relative-humidity fraction for values below 2, otherwise
            specific humidity in grams per kilogram.
        wind_height_m: Wind measurement height in metres.
        temperature_height_m: Temperature and humidity measurement height in
            metres.
        pressure_mb: Surface pressure in millibars.
        boundary_layer_height_m: Marine boundary-layer height in metres.
        shortwave_down_w_m2: Downward shortwave radiation in watts per square
            metre.
        longwave_down_w_m2: Downward longwave radiation in watts per square
            metre.
        rainfall_mm_h: Rainfall rate in millimetres per hour.
        wave_method: Roughness method: 0 uses Charnock, 1 uses wave age, and 2
            uses wave slope.
        wave_period_s: Wave period in seconds; a negative value derives it
            from wind speed for wave methods 1 and 2.
        wave_height_m: Wave height in metres; a negative value derives it from
            wind speed for wave methods 1 and 2.
        coastal_shelf_length_km: Shelf adjustment length in kilometres; values
            below 0.1 disable the adjustment.
        distance_to_coast_km: Water-cell distance to land in kilometres.
        cool_skin: Whether to calculate the cool-skin temperature correction.

    Returns:
        COARE flux arrays with the common broadcast shape and ``float32``
        dtype.

    Raises:
        ValueError: If ``wave_method`` is outside 0--2, or a measurement
            height, pressure, or boundary-layer height is non-positive.
    """
    if wave_method not in {0,1,2}:
        raise ValueError('COARE wave_method must be 0, 1, or 2')
    u,t_k,ts_k,q_input=np.broadcast_arrays(
        np.asarray(wind_speed_m_s,dtype=MODEL_DTYPE),
        np.asarray(air_temperature_k,dtype=MODEL_DTYPE),
        np.asarray(sea_temperature_k,dtype=MODEL_DTYPE),
        np.asarray(humidity,dtype=MODEL_DTYPE))
    zu,zt,p,zi,rs,rl,rain=np.broadcast_arrays(
        np.asarray(wind_height_m,dtype=MODEL_DTYPE),
        np.asarray(temperature_height_m,dtype=MODEL_DTYPE),
        np.asarray(pressure_mb,dtype=MODEL_DTYPE),
        np.asarray(boundary_layer_height_m,dtype=MODEL_DTYPE),
        np.asarray(shortwave_down_w_m2,dtype=MODEL_DTYPE),
        np.asarray(longwave_down_w_m2,dtype=MODEL_DTYPE),
        np.asarray(rainfall_mm_h,dtype=MODEL_DTYPE),
        subok=True)
    if np.any(zu<=0.) or np.any(zt<=0.) or np.any(p<=0.) or np.any(zi<=0.):
        raise ValueError('COARE heights and pressure must be positive')
    t=t_k-273.15; ts=ts_k-273.15
    to_k=273.16; gravity=9.81; von=.4; fdg=1.; beta=1.2
    cpa=1004.67; cpw=4000.; rhow=1022.; rgas=287.1; visw=1.e-6
    expansion=2.1e-5*np.power(ts+3.2,.79)
    latent=(2.501-.00237*ts)*1.e6
    viscosity=1.326e-5*(1.+6.542e-3*t+8.301e-6*t*t-4.84e-9*t*t*t)
    saturated_air=_humidity_vapor_pressure(t,p)
    vapor=np.where(q_input<2.,saturated_air*q_input,np.nan)
    q=np.where(q_input<2.,.62197*vapor/(p-.378*vapor),q_input/1000.)
    # ASL recomputes density after humidity has been converted.
    rho_air=p*100./(rgas*(t+to_k)*(1.+.61*q))
    saturated_sea=.98*_humidity_vapor_pressure(ts,p)
    qs=.62197*saturated_sea/(p-.378*saturated_sea)
    rnl=.97*(5.67e-8*(ts+to_k)**4-rl)
    rns=.945*rs
    salinity=.026
    conductivity=.6
    big_c=(16.*gravity*cpw*(rhow*visw)**3/
           (conductivity*conductivity*rho_air*rho_air))
    wet_c=.622*latent*qs/(rgas*(ts+to_k)**2)
    dter=np.where(cool_skin,.3,0.)
    gust=np.full_like(u,.5)
    speed=np.sqrt(u*u+gust*gust)
    delta_t=ts-t-.0098*zt
    delta_q=qs-q
    roughness=np.full_like(u,1.e-4)
    u10=speed*np.log(10./roughness)/np.log(zu/roughness)
    ustar=.035*u10
    zo10=.011*ustar*ustar/gravity+.11*viscosity/ustar
    cd10=(von/np.log(10./zo10))**2
    ct10=.00115/np.sqrt(cd10)
    zot10=10./np.exp(von/ct10)
    cd=(von/np.log(zu/zo10))**2
    ct=von/np.log(zt/zot10)
    coefficient=von*ct/cd
    rib_critical=-zu/(zi*.004*beta**3)
    rib=-gravity*zu*((delta_t-dter)+.61*(t+to_k)*delta_q)/(
        (t+to_k)*speed*speed)
    zetu=np.where(rib<0.,coefficient*rib/(1.+rib/rib_critical),
                  coefficient*rib*(1.+3.*rib/coefficient))
    # The original performs one stability iteration for all cases.
    tscale=-(delta_t-dter)*von*fdg/(np.log(zt/zot10)-_psit(zetu))
    qscale=-(delta_q-wet_c*dter)*von*fdg/(np.log(zt/zot10)-_psit(zetu))
    ustar=speed*von/(np.log(zu/zo10)-_psiu(zetu))
    charn=np.where(speed>18.,.018,
                   np.where(speed>10.,.011+(.018-.011)*(speed-10.)/8.,.011))
    shelf=np.asarray(coastal_shelf_length_km,dtype=MODEL_DTYPE)
    coast=np.asarray(distance_to_coast_km,dtype=MODEL_DTYPE)
    shelf_safe=np.where(shelf<.1,1.,shelf)
    factor=np.where(shelf<.1,1.,1.-(1.-.032/charn)*np.exp(-(coast/shelf_safe)**4))
    charn=charn*factor
    ta=t+to_k
    zet=von*gravity*zu/ta*(tscale+.61*ta*qscale)/(ustar*ustar)
    if wave_method==0:
        roughness=charn*ustar*ustar/gravity+.11*viscosity/ustar
    else:
        period_input=np.asarray(wave_period_s,dtype=MODEL_DTYPE)
        height_input=np.asarray(wave_height_m,dtype=MODEL_DTYPE)
        period=np.where(period_input<0.,.729*u,period_input)
        height=np.where(height_input<0.,.018*u*u*(1.+.015*u),height_input)
        cwave=gravity*period/(2.*np.pi); lwave=cwave*period
        if wave_method==1:
            roughness=(50./(2.*np.pi))*lwave*(ustar/cwave)**4.5+.11*viscosity/ustar
        else:
            roughness=1200.*height*(height/lwave)**4.5+.11*viscosity/ustar
    reynolds=roughness*ustar/viscosity
    scalar_roughness=np.minimum(1.15e-4,5.5e-5/reynolds**.6)
    ustar=speed*von/(np.log(zu/roughness)-_psiu(zet))
    tscale=-(delta_t-dter*bool(cool_skin))*von*fdg/(
        np.log(zt/scalar_roughness)-_psit(zet))
    qscale=-(delta_q-wet_c*dter*bool(cool_skin))*von*fdg/(
        np.log(zt/scalar_roughness)-_psit(zet))
    buoyancy=-gravity/ta*ustar*(tscale+.61*ta*qscale)
    gust=np.where(buoyancy>0.,beta*np.cbrt(np.maximum(buoyancy*zi,0.)),.2)
    sensible=-cpa*rho_air*ustar*tscale
    latent_flux=-latent*rho_air*ustar*qscale
    # ASL computes the cool-skin correction after the returned scales.  Keep
    # that state for completeness even though it affects the next call only.
    heat_out=rnl+sensible+latent_flux
    thickness=np.full_like(u,.001)
    absorbed=rns*(.060+11.*thickness-6.6e-5/thickness*(1.-np.exp(-thickness/8.e-4)))
    column=heat_out-absorbed
    alpha_q=expansion*column+salinity*latent_flux*cpw/latent
    ratio=big_c*np.abs(alpha_q)/ustar**4
    positive=np.maximum(big_c*alpha_q/ustar**4,0.)
    lam=np.where(alpha_q>0.,6./np.cbrt(1.+positive**.75),
                 np.where(ratio<1.,6./np.cbrt(np.maximum(1.-ratio**.75,1.e-30)),6.))
    thickness=lam*visw/(np.sqrt(rho_air/rhow)*ustar)
    dter=column*thickness/conductivity
    total_speed=np.sqrt(u*u+gust*gust)
    stress=rho_air*ustar*ustar*u/total_speed
    saturated_specific=.62197*saturated_air/(p-.378*saturated_air)
    water_diffusivity=2.11e-5*((t+to_k)/to_k)**1.94
    heat_diffusivity=(1.+3.309e-3*t-1.44e-6*t*t)*.02411/(rho_air*cpa)
    saturation_slope=saturated_specific*latent/(rgas*(t+to_k)**2)
    wet_bulb=1./(1.+.622*saturation_slope*latent*water_diffusivity/(cpa*heat_diffusivity))
    rain_heat=(rain*wet_bulb*cpw*((ts-t-dter*bool(cool_skin))+
               (qs-q-wet_c*dter*bool(cool_skin))*latent/cpa)/3600.)
    z_over_l=zet
    monin=np.where(np.abs(z_over_l)<1.e-4,10000.,zu/z_over_l)
    buoyancy_flux=-ustar*(tscale+.61*qscale*t_k)
    return CoareFlux(*(np.asarray(value,dtype=MODEL_DTYPE) for value in (
        roughness,monin,ustar,tscale,qscale,sensible,latent_flux,
        buoyancy_flux,dter,gust,stress,rain_heat)))
