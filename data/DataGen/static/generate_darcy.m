function generate(N, S)
    % Set default values if not provided
    % for round = 1:5
    for round = 6
        rng(10000000 + round, "twister");
        if nargin < 1
            N = 10000; % Default number of generations
        end
        if nargin < 2
            S = 128; % Default resolution
        end
    
        % Preallocate arrays to store the generated data
        lognorm_a_data = zeros(N, S, S);
        thresh_a_data = zeros(N, S, S);
        lognorm_p_data = zeros(N, S, S);
        thresh_p_data = zeros(N, S, S);
    
        % Parameters for Gaussian Random Field (GRF)
        % alpha = 2;
        % tau = 3;
        alpha = 3;
        tau = 4;
    
        % Forcing function, f(x) = 1
        f = ones(S, S);

        t0 = tic;
        updateEvery = max(1, floor(N / 100));
    
        for i = 1:N
            % Generate random coefficients from N(0,C)
            norm_a = GRF(alpha, tau, S);
    
            % Exponentiate it to ensure a(x) > 0 (Lognormal)
            lognorm_a = exp(norm_a);
    
            % Thresholding to achieve ellipticity
            thresh_a = zeros(S, S);
            thresh_a(norm_a >= 0) = 12;
            thresh_a(norm_a < 0) = 4;
    
            % Solve PDE: -div(a(x)*grad(p(x))) = f(x)
            lognorm_p = solve_gwf(lognorm_a, f);
            thresh_p = solve_gwf(thresh_a, f);
    
            % Store the generated data
            lognorm_a_data(i, :, :) = lognorm_a;
            thresh_a_data(i, :, :) = thresh_a;
            lognorm_p_data(i, :, :) = lognorm_p;
            thresh_p_data(i, :, :) = thresh_p;

            if mod(i, updateEvery) == 0 || i == N
                elapsed = toc(t0);
                eta = elapsed * (N - i) / i;

                fprintf('\rProgress [Darcy]: %6.2f%%  [%d/%d]  elapsed: %.1fs  ETA: %.1fs', ...
                    100 * i / N, i, N, elapsed, eta);
            end
        end
    
        % Ensure the data folder exists
        if ~exist('/large_storage/zhangxf/PDEdata/darcy/', 'dir')
           mkdir('/large_storage/zhangxf/PDEdata/darcy/')
        end
    
        % Save the data in a .mat file
        % filename = sprintf('/large_storage/zhangxf/PDEdata/darcy/darcy_%d-%d-%d_%d.mat', N, S, S, round);
        filename = sprintf('/large_storage/zhangxf/PDEdata/darcy/darcy_test_%d-%d-%d.mat', N, S, S);
        % filename = sprintf('/large_storage/zhangxf/PDEdata/darcy/darcy_%d-%d-%d_test.mat', N, S, S);
        save(filename, 'lognorm_a_data', 'thresh_a_data', 'lognorm_p_data', 'thresh_p_data', '-v7.3');
    end
end
