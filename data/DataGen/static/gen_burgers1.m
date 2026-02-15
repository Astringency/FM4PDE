for round = 1:5
    sprintf("Init Round %d", round)
    % number of realizations to generate
    N = 10000;
    
    % parameters for the Gaussian random field
    gamma = 2.5;
    tau = 7;
    sigma = 7^(2);
    
    % viscosity
    visc = 1/100;
    
    % grid size
    s = 128;
    steps = 127;
    
    
    input = zeros(N, s);
    if steps == 1
        output = zeros(N, s);
    else
        output = zeros(N, steps, s);
    end
    
    tspan = linspace(0,1,steps+1);
    x = linspace(0,1,s+1);

    h = waitbar(0, 'Generating...');

    for j=1:N
        waitbar(j/N, h, sprintf('Generating: %d', j));

        u0 = GRF1(s/2, 0, gamma, tau, sigma, "periodic");
        u = burgers1(u0, tspan, s, visc);
        
        u0eval = u0(x);
        input(j,:) = u0eval(1:end-1);
        
        if steps == 1
            output(j,:) = u.values;
        else
            for k=2:(steps+1)
                output(j,k,:) = u{k}.values;
            end
        end
        
        output(j,1,:)=input(j,:);
    
    end
    delete(h);
    
    sprintf("Saving (Round %d)", round)

    output = output(:, 1:end, :);
    tspan = tspan(1:end);
    x = x(1:end);

    filename = sprintf('/large_storage/zhangxf/PDEdata/burgers/burger_%d-%d-%d_%d.mat', N, s, steps+1, round);
    save(filename, 'output', 'input');

end
