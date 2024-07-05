function [u_sc, u_sc_broaden, A, A_broaden] = gen_system(G, u_hat_inc_diag, contrast_vec)
A = G*u_hat_inc_diag;
u_sc = A*contrast_vec; %u_sc
A_broaden = [real(A); imag(A)];
u_sc_broaden = A_broaden*contrast_vec;
end