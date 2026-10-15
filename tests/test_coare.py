import numpy as np

from pycalmet.coare import (_psit,_psiu,advance_warm_layer,distance_to_coast,
                           downward_longwave,initial_warm_layer_state,surface_flux,
                           update_warm_layer_fluxes)


def test_coare_stability_functions_cover_stable_and_unstable_branches():
    values=np.array([-2.,-.1,0.,1.,100.])
    np.testing.assert_allclose(_psiu(values),
        [1.5323452,.27006426,-.004999161,-4.286161,-109.525],
        rtol=2.e-6,atol=1.e-7)
    np.testing.assert_allclose(_psit(values),
        [2.3973062,.5112704,-.004999161,-4.4378185,-565.1493],
        rtol=2.e-6,atol=1.e-7)
    assert _psiu(values).dtype==np.float32
    assert _psit(values).dtype==np.float32


def test_coare_stability_functions_do_not_evaluate_inactive_stable_branch():
    strongly_unstable=np.array([-10000.,-1000.,-100.],dtype=np.float32)
    with np.errstate(over='raise',divide='raise',invalid='raise'):
        momentum=_psiu(strongly_unstable)
        temperature=_psit(strongly_unstable)
    assert np.isfinite(momentum).all()
    assert np.isfinite(temperature).all()


def test_coare_scalar_uses_standard_constants_consistently():
    result=surface_flux(5.,288.15,290.15,.8,boundary_layer_height_m=600.,
                        shortwave_down_w_m2=300.,longwave_down_w_m2=350.)
    expected=[4.2853629565797746e-5,-22.04370880126953,
              .17281965911388397,-.07310386747121811,
              -.00012885473552159965,15.38531494140625,
              66.42070007324219,.016547976061701775]
    actual=[result.roughness_m,result.monin_obukhov_m,result.friction_velocity_m_s,
            result.temperature_scale_k,result.humidity_scale_kg_kg,
            result.sensible_heat_flux_w_m2,result.latent_heat_flux_w_m2,
            result.buoyancy_flux_k_m_s]
    np.testing.assert_allclose(actual,expected,rtol=2.e-6,atol=2.e-12)


def test_coare_vectorizes_grid_and_wave_methods():
    wind=np.array([[2.,5.],[10.,15.]])
    for method in (0,1,2):
        result=surface_flux(wind,np.full_like(wind,288.15),np.full_like(wind,290.15),.8,
                            wave_method=method,wave_period_s=-999.,wave_height_m=-999.)
        assert result.roughness_m.shape==wind.shape
        assert all(np.isfinite(field).all() for field in (
            result.roughness_m,result.monin_obukhov_m,result.friction_velocity_m_s,
            result.temperature_scale_k,result.humidity_scale_kg_kg,
            result.sensible_heat_flux_w_m2,result.latent_heat_flux_w_m2,
            result.buoyancy_flux_k_m_s))


def test_coare_warm_layer_carries_flux_state_and_resets_at_midnight():
    state=initial_warm_layer_state((1,2))
    sea=np.full((1,2),300.)
    adjusted,state,warming=advance_warm_layer(
        state,np.full((1,2),8.*3600.),sea,.6,700.,350.)
    np.testing.assert_array_equal(adjusted,sea)
    np.testing.assert_array_equal(warming,0.)
    flux=surface_flux(3.,298.,sea,.8,shortwave_down_w_m2=700.,
                      longwave_down_w_m2=350.)
    state=update_warm_layer_fluxes(state,flux)
    adjusted,state,warming=advance_warm_layer(
        state,np.full((1,2),9.*3600.),sea,.6,700.,350.)
    assert np.all(state.morning_started)
    assert np.all(state.stress_integral_n_s_m2>0.)
    assert np.all(state.heat_column_j_m2>0.)
    assert np.all(warming>0.)
    assert np.all(adjusted>sea)
    reset,state,warming=advance_warm_layer(
        state,np.full((1,2),1.*3600.),sea,.6,0.,350.)
    np.testing.assert_array_equal(reset,sea)
    np.testing.assert_array_equal(warming,0.)
    np.testing.assert_array_equal(state.heat_column_j_m2,0.)
    np.testing.assert_array_equal(state.stress_integral_n_s_m2,0.)
    assert not state.morning_started.any()


def test_coare_warm_layer_remains_numerically_close_to_reference():
    expected=np.array([
        [2.8727345679896709e-5,-6.2816214561462402,.10695283632798315,
         -.08312359478532598,-.0002636701513303671,10.423057265678789,80.21008861204011],
        [2.8733348263428452e-5,-6.208354949951172,.10702416324580075,
         -.08448439940081932,-.00026558822387876614,10.600756602250772,80.84516174820342],
        [2.8733926461990408e-5,-6.201370716094971,.10703102144303901,
         -.08461589930740357,-.0002657736225678791,10.617937034860315,80.90655937774518],
        [2.8734377942921357e-5,-6.195926189422607,.10703637510024781,
         -.08471863123925903,-.00026591846791866326,10.631359993769232,80.9545286358173],
    ])
    state=initial_warm_layer_state(())
    actual=[]
    for hour in range(4):
        sea,state,_=advance_warm_layer(
            state,(120./15.+hour)*3600.,300.,.6,700.,350.)
        flux=surface_flux(3.,298.,sea,.8,boundary_layer_height_m=600.,
                          shortwave_down_w_m2=700.,longwave_down_w_m2=350.)
        state=update_warm_layer_fluxes(state,flux)
        actual.append([flux.roughness_m,flux.monin_obukhov_m,
                       flux.friction_velocity_m_s,flux.temperature_scale_k,
                       flux.humidity_scale_kg_kg,flux.sensible_heat_flux_w_m2,
                       flux.latent_heat_flux_w_m2])
    # The modern path uses exact cube roots and pi rather than shortened
    # constants; it remains close to the original CALMET reference values.
    np.testing.assert_allclose(actual,expected,rtol=5.e-5,atol=2.e-9)


def test_radflx_and_setcoast_follow_calmet_grid_definitions():
    temperature=np.array([[290.,290.,290.]])
    cloud=np.array([[.5,.5,.5]])
    ceiling=np.array([[1000.,3000.,7000.]])
    got=downward_longwave(temperature,50.,cloud,ceiling)
    water=np.clip(.439*.5*np.exp(26.23-5416./290.)/290.,.1,5.)
    clear=(.725+.17*np.log10(water))*5.67e-8*290.**4
    np.testing.assert_allclose(got,clear*(1.+cloud*np.array([[.26,.22,.006]])))
    landuse=np.array([[1,55,55],[1,55,55],[1,1,55]])
    expected=np.array([[0.,2.,4.],[0.,2.,np.sqrt(13.)],[0.,0.,2.]])
    np.testing.assert_allclose(distance_to_coast(landuse,(55,55),2.,3.),expected)
