function [fitresult, gof] = createFit_exp2(binWidth, counts, plot_flag)

% Notes
% gof — Goodness-of-fit statistics
% gof structure
% sse:        Sum of squares due to error
% rsquare:    R-squared (coefficient of determination)
% dfe:        Degrees of freedom in the error
% adjrsquare: Degree-of-freedom adjusted coefficient of determination
% rmse:       Root mean squared error (standard error)

% Prepare data for fitting
t               = (0:numel(counts)-1)*binWidth;                 % in [ns]

% Use MatLab prepareCurveData routine
[xData, yData]  = prepareCurveData( t, double(counts) );

% Set up fittype and options
%ft                 = fittype( 'exp1' );
ft                  = fittype( 'a*exp(-x/tau)+c' );
opts                = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Algorithm      = 'Levenberg-Marquardt';
opts.Display        = 'Off';
opts.MaxFunEvals    = 10000;
opts.MaxIter        = 10000;


% Fit model to data
[fitresult, gof] = fit( xData, yData, ft, opts );

% Plot fit with data
if plot_flag == 1
    figure( 'Name', 'Single Exp Fit' );
    plot( fitresult, xData, yData );
    title(sprintf('Fitting to function: a * exp(-t/tau) + c\ntau = %3.2f, a = %f, c = %f\n(rsquare = %f & rmse = %e)', ...
        fitresult.tau, fitresult.a, fitresult.c, gof.rsquare, gof.rmse))
    xlabel('t [ns]')
    ylabel Counts
    grid on
    savefig(gcf, 'decayExpFit.fig');
    saveas (gcf, 'decayExpFit.png');
end