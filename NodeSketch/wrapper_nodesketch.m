function wrapper_nodesketch(in_file, out_file, K_hash, order, alpha)
    try
        disp(['Loading ', in_file]);
        data = load(in_file);
        network = data.A; 
        network = sparse(double(network));
        num_nodes = size(network, 1);
        
        disp('Generating Random beta...');
        Rand_beta = -log(rand(K_hash, num_nodes));
        
        disp(['Running NodeSketch scalable (K=', num2str(K_hash), ', order=', num2str(order), ')...']);
        embs = sketch_node_embs_scalable(network + speye(num_nodes), K_hash, Rand_beta, order, alpha);
        
        disp(['Saving to ', out_file]);
        save('-v7', out_file, 'embs');
    catch e
        disp('Error in NodeSketch execution:');
        disp(e.message);
        exit(1);
    end
    exit(0);
end
