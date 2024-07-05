function G = gen_G(a, kb, M, n, x_rec_grid, y_rec, h)
G = [];
G_blk = zeros(M, n);
for blk_id = 1:n
    for y = 1:M
        for x = 1:n
            rho = norm([x_rec_grid(y)-x*h, y_rec-y*h]);
            if rho == 0
                G_blk(y,x) = -0.5i/(kb*a) * (besselh(1, 2, kb*a) - 2i/(pi*kb*a));
            else
                G_blk(y,x) = -1i/(2*kb*a) * besselj(1,kb*a)*besselh(0, 2, kb*rho);
            end
        end
    end
    G = [G, G_blk];
end
end