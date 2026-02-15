function generate_GRF(N, S)
    for round = 1:5
        if nargin < 1
            N = 5000; % Default number of generations
        end
        if nargin < 2
            S = 64; % Default resolution
        end

        f_data = zeros(N, S, S);
        
        % Parameters for GRF
        alpha = 2;
        tau = 3;
        
        hbar = waitbar(0, 'Generating...');

        for i = 1:N
            waitbar(i/N, hbar, sprintf('Generating Test f data: %d', i));
            f = GRF(alpha, tau, S);
            f_data(i, :, :) = f;
        end
        delete(hbar);
        % Save the dataset
        if ~exist('data', 'dir')
            mkdir('data');
        end
        filename = sprintf('testdata/testfdata_%d-%d-%d_%d.mat', N, S, S, round);
        save(filename, 'f_data');
    end
end
