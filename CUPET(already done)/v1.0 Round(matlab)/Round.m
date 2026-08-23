clear
rho=imread('C:\Users\huang\Desktop\afa8815267f77df76dc71b80fc1deab6.jpg');
imshow(rho)
set(gcf,'outerposition',get(0,'screensize'));
hold on
[p1,p2] = ginput(3);
x=[p1(1),p2(1)];
y=[p1(2),p2(2)];
linex=[x(1),y(1)];
liney=[x(2),y(2)];
plot(linex,liney,'r-','Linewidth',3);
hold on
z=[p1(3),p2(3)];
A=[x(1)-y(1),x(2)-y(2);z(1)-y(1),z(2)-y(2)];
B=[x(1)^2-y(1)^2+x(2)^2-y(2)^2;z(1)^2-y(1)^2+z(2)^2-y(2)^2];
ab=A\B;
a=ab(1)/2;
b=ab(2)/2;
circleCenter = [a,b];
c2 = (x(1)-a)^2+(x(2)-b)^2;
radius = sqrt(c2);
theta = (0:pi/360:2*pi)';
Result = zeros(size(theta,1),4);
for i = 1: size(theta,1)
Result(i,1) = i;
Result(i,2) = theta(i);
Result(i,3) = circleCenter(1) + radius*cos(theta(i));
Result(i,4) = circleCenter(2) + radius*sin(theta(i));
end
plot(Result(:,3),Result(:,4),'r-','Linewidth',3);
hold on
ca_left=90-asin((circleCenter(2)-x(2))/radius)/3.1415926*180;
ca_right=90-asin((circleCenter(2)-y(2))/radius)/3.1415926*180;
