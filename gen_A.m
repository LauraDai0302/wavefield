function A_output= gen_A (kb_input,grid_points_x,grid_points_y,rho_s_x,rho_s_y,rec_points_x,rec_points_y,h)
u_hat_inc = - 1i/4* besselh(0,2,kb_input*abs((grid_points_x-rho_s_x)+1i*(grid_points_y-rho_s_y)));

u_hat_inc_vec = reshape(u_hat_inc,[],1);
A_output = [];
for ii = 1:length(rec_points_x)
temp = - 1i/4* besselh(0,2,kb_input*abs((grid_points_x-rec_points_x(ii))+1i*(grid_points_y-rec_points_y(ii))));
temp = reshape(temp,[],1)';
temp = kb_input.^2.*h.*h.*temp;
A_output = [A_output; temp];
end
A_output = A_output*diag(u_hat_inc_vec);
end