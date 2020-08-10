x=-10:0.1:10;  % Test values.
s = 1./(1+exp(-x)); % Derivative of sigmoid.
ds1 = s.*(1-s); % Another simpler way to compute the derivative of a sigmoid.
dds = 2*s.^3 - 3*s.^2 + s;
figure; plot(x,s,'r+'); hold on; 
plot(x,ds1, 'go'); 
plot(x,dds, 'b*'); 