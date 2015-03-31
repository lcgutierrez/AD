%
% Test Rosenbrock function
%

f = @(x) (1-x(1))^2+100*(x(2)-x(1)^2)^2;
x=AD([1,2]);
y=f(x);
disp(y);

%
% Test Ackley function
%
f = @(x) -20*exp(-0.2*sqrt(0.5*(x(1)^2+x(2)^2))) ...
            -exp(0.5*(cos(2*pi*x(1))+cos(2*pi*x(2))))+20+exp(1);
x = AD([1,2]);
y=f(x);
disp(y);
