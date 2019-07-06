clc;
close all;
clear all;

 I1=imread('D:\Petnica projekat\tumor library - Copy\distal femur\456-2 - Copy (2).jpg')
 %I1=rgb2gray(I1);
 I=double(I1);
 %I=imresize(I,[512 512]);
[row col] = size(I);
loc_wind_size=10;
mean=0,001;
a=0.5
b=0.9
t1=1
t2=100
beta=0.2


  del_sort1= zeros(row,col);
  del_sort2= zeros(1,col);
 u2= zeros(1,col);
 u3= zeros(1,col);
 del_nw_xy_f= 0;
 del_n_xy_f=0;
 del_ne_xy_f=0;
 del_e_xy_f=0;
 del_se_xy_f=0;
 del_s_xy_f=0;
 del_sw_xy_f=0;
 del_w_xy_f=0;
            
    for i=4:row-3    
        for j=4:col-3
                del_nw_xy_f=I(i-1,j-1);
                del_n_xy_f=I(i-1,j);
                del_ne_xy_f=I(i-1,j+1);
                del_e_xy_f=I(i,j+1);
                del_se_xy_f=I(i+1,j+1);
                del_s_xy_f=I(i+1,j);
                del_sw_xy_f=I(i+1,j-1);
                del_w_xy_f=I(i,j-1); 
        
       del_sort = abs(del_nw_xy_f - del_se_xy_f) + abs(del_s_xy_f - del_n_xy_f) + abs(del_ne_xy_f - del_sw_xy_f ) + abs(del_w_xy_f - del_e_xy_f)
       %del_sort=[del_n_xy_f,del_nw_xy_f,del_e_xy_f,del_se_xy_f,del_s_xy_f,del_sw_xy_f,del_w_xy_f,del_ne_xy_f];  
       %del_sort=sort(del_sort);
       del_sort = del_sort \ 255
       del_sort1(i,j)=del_sort;
       if(del_sort1(i,j)==0)
            del_sort1(i,j)=1;
       end
       del_sort2(j)=del_sort;
        end
    end
       
figure;imshow(I); 
figure;imshow(del_sort1);

[t1,t2,beta,a,b]=gsa(I,del_sort2);
a;
b;
t1;
t2;
beta;
    
for j=4:col-3
u1(j)=1/(1+abs((del_sort2(j)-t1)/t2)^2*beta);  
       if(u1(j)<=a)
          u2(j)=0;
      elseif((u1(j)>a)&&(u1(j)<b))
          u2(j)=abs((u1(j)-a)/(a-b));
      elseif(u1(j)>=b)
          u2(j)=1;
       end
end
     
i2=zeros(430,512);
for i=4:row-3  
    for j=4:col-3
        for k=4:col-3
        if  (del_sort1(i,j)==del_sort2(k))
            i2(i,j)=u2(k);
        end        
    end
    end
end
figure;imshow(i2);

i3=i2*255;
figure;imshow(i3);

i4=adaptivethreshold(i3,loc_wind_size,mean,0);
figure;imshow(i4);

i5=i4;
i6=1-i5;
[row1 col1]= size(i5);
for i=5:row1-5
for j=5:col1-5
a=i6(i+1,j);
b=i6(i-1,j);
c=i6(i,j+1);
d=i6(i,j-1);
sum=a+b+c+d;
if(sum<1)
i6(i,j)=0;
end
end
end
i9=1-i6;
i10=1-i9;
[row1 col1]= size(i9);
for i=5:row1-5
for j=5:col1-5
a=i10(i+1,j);
b=i10(i-1,j);
c=i10(i,j+1);
d=i10(i,j-1);
sum=a+b+c+d;
if(sum<1)
i10(i,j)=0;
end
end
end
i13=1-i10;
i14=1-i13;
[row1 col1]= size(i13);
for i=5:row1-5
for j=5:col1-5
a=i14(i+1,j);
b=i14(i-1,j);
c=i14(i,j+1);
d=i14(i,j-1);
sum=a+b+c+d;
if(sum<1)
i14(i,j)=0;
end
end
end
i15=1-i14;
figure;imshow(i15);
