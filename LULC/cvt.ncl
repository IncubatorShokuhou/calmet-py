begin

a=asciiread("GLC2000CN.txt",(/6429,8170/),"integer")
printVarSummary(a)
print(a(0,0))

lat=new((/6429/),"double")  
lon=new((/8170/),"double")  

lat(0)=0.0044732855999996
do i=1,6428
lat(i)=lat(0)+i*0.00892857
end do

lon(0)=68.0937497828
do i=1,8169
lon(i)=lon(0)+i*0.00892857
end do

data=new((/8170*6429,4/),"double")  
n=1
do i=0,6428
do j=0,8169
data(8170*i+j,0)=n
data(8170*i+j,1)=a(i,j)
data(8170*i+j,2)=lat(i)
data(8170*i+j,3)=lon(j)
n=n+1
end do
end do

filename = "lu.csv"
  system("rm -rf " + filename)
  write_table(filename,"w",[/data(:,0),data(:,1),data(:,2),data(:,3)/],"%2i,%2i,%20.16f,%20.16f")


end