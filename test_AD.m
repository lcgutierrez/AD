f = @(x) (1-x(1))^2+100*(x(2)-x(1)^2)^2;
x=AD([1,2]);
f(x)