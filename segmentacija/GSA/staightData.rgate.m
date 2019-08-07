function [fxys2, fxys] = staightData(img, polyDegree)
%make an input membrane black and white image straight
%Syntax: xy = staightData(img, polyDegree)
%outptu xy is (N, 2) matrix
%polyDegree is optional degree of fitted polynom, default 5

PLOT_OUTPUTS = 1;
%note that plotted fy values are up-and-down flipped with comparison to the input image

if nargin < 2
    polyDegree = 5;
end

%step 1 - get x and y coordinates
[fy, fx] = find(img == 1); %also thresholding of grayscale image can do the same job
if PLOT_OUTPUTS
   figure, plot(fx, fy, '.') 
   title('Input pixels')
end

%step 2 - rotate image to be "left to right" horizontal - to ensure successful polyfit
c = cov([fx fy]);
[V, L] = eig(c);
[~, idx] = sort(diag(L), 'descend');
fxy = [fx fy] * V(:, idx);
if PLOT_OUTPUTS
   figure, plot(fxy(:, 1), fxy(:,2), 'k.') 
   title('Horizontal rotation of input data')
end

%step 3 - polynomial fit of the data
p = polyfit(fxy(:,1), fxy(:,2), polyDegree);
if PLOT_OUTPUTS
    xx = (min(fxy(:,1)):max(fxy(:,1)))';
    yy = polyval(p, xx);
    figure, plot(fxy(:, 1), fxy(:,2), '.')
    title(sprintf('Polynomial fit (degree %d)', polyDegree));
    hold on
    plot(xx, yy, 'r')
    hold off
end

%step 4 - subtration of the polyfit
fxys = fxy;
fxys(:, 2) = fxys(:, 2) - polyval(p, fxy(:, 1));
if PLOT_OUTPUTS
    figure, plot(fxys(:, 1), fxys(:,2), 'r.')
    title('Final wiggles after subtraction of the polynomial fit')
end

%step 5 - a normal distance from the polyfit
fxys2 = fxy;
%generate fine-polynomial values
xx = (min(fxy(:,1)) - 50:.25:max(fxy(:,1)) + 50)';
yy = polyval(p, xx);
T = [0 1; -1 0];
dp = diff([xx yy]) * T; %comoute normal from first derivation of the poly
cp = [0; cumsum( sqrt( sum(dp.^2, 2) ) )]; %calculate cummulative distance according to the poly to be a new x axis values
dp = [dp(1, :); dp];

for k = 1:length(fx)
    [mn, ma] = min( sum( ( [xx yy] - repmat(fxy(k,:), size(xx, 1), 1) ) .^2, 2) ); %find minimal distance point of the poly
    d = sqrt(mn) * sign(dp(ma, :) * (fxy(k, :) - [xx(ma) yy(ma)])'); %evaluate distance and add signum
    fxys2(k, 1) = cp(ma);
    fxys2(k, 2) = d;
end
if PLOT_OUTPUTS
    figure, plot(fxys2(:, 1), fxys2(:,2), 'm.')
    title('Final wiggles after calculation of a normal distance from the polynomial fit')
end


img = '524 pagetic sarcoma'
polydegree = 5
[fxys2, fxys] = staightData(img, polyDegree)
