function profile = get_generation_profile(pde, dataset_type)
%GET_GENERATION_PROFILE Return the explicit GRF profile for a dataset type.

    canonical_type = canonicalize_dataset_type(dataset_type);
    pde = lower(char(pde));

    switch canonical_type
        case 'train'
            seed_offset = 0;
        case 'id'
            seed_offset = 10000000;
        case 'smooth'
            seed_offset = 20000000;
        case 'rough'
            seed_offset = 30000000;
        otherwise
            error('Unsupported dataset type: %s', canonical_type);
    end

    switch pde
        case {'poisson', 'helmholtz', 'darcy'}
            switch canonical_type
                case 'train'
                    alpha = 2.0;
                    tau = 3.0;
                case 'id'
                    alpha = 2.0;
                    tau = 3.0;
                case 'smooth'
                    alpha = 3.0;
                    tau = 4.0;
                case 'rough'
                    alpha = 1.5;
                    tau = 5.0;
            end
            profile = struct( ...
                'dataset_type', canonical_type, ...
                'alpha', alpha, ...
                'tau', tau, ...
                'seed_offset', seed_offset);

        case {'burger', 'burgers', 'nsnonbounded'}
            switch canonical_type
                case 'train'
                    alpha = 2.5;
                    tau = 7.0;
                case 'id'
                    alpha = 2.5;
                    tau = 7.0;
                case 'smooth'
                    alpha = 3.0;
                    tau = 6.5;
                case 'rough'
                    alpha = 1.5;
                    tau = 5.0;
            end
            profile = struct( ...
                'dataset_type', canonical_type, ...
                'alpha', alpha, ...
                'tau', tau, ...
                'seed_offset', seed_offset);

        otherwise
            error('No generation profile is defined for PDE: %s', pde);
    end
end


function canonical_type = canonicalize_dataset_type(dataset_type)
    dataset_type = lower(char(dataset_type));
    switch dataset_type
        case 'train'
            canonical_type = 'train';
        case {'id', 'test'}
            canonical_type = 'id';
        case {'easytest', 'easy', 'smooth'}
            canonical_type = 'smooth';
        case {'hardtest', 'hard', 'rough'}
            canonical_type = 'rough';
        otherwise
            error(['dataset_type must be train, id, smooth, or rough ' ...
                   '(legacy aliases easytest and hardtest are accepted).']);
    end
end
