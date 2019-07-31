import pyextrapolation.constant.extrapolator as extrapolator
import numpy as np
from skimage.draw import circle
from scipy.ndimage.morphology import distance_transform_edt
import matplotlib.pyplot as plt


class test_CE:

    def construct_random(self, ndims):
        shape = []

        for _ in range(ndims):
            shape.append(100)

        u = np.random.standard_normal(size=tuple(shape))
        phi = np.random.standard_normal(size=tuple(shape))

        my_ce_obj = extrapolator.CE(u, phi)

        return my_ce_obj

    def test_constructor(self):
        print('\n\n\tRunning test_constructor...')

        for ndims in (1, 2, 3):

            my_ce_obj = self.construct_random(ndims)

            assert my_ce_obj.u.shape == my_ce_obj.phi.shape
        print('\tTest complete.')

    def test_solve_random(self):
        print('\n\n\tRunning test_solve_random...')

        for ndim in (1, 2, 3):

            my_ce_obj = self.construct_random(ndim)
            u_n = my_ce_obj.solve()
            print('\tSolve complete for ndim =', ndim)
        print('\tTest complete.')

    def test_upwind_differences_1D(self):
        print('\n\n\tRunning test_upwind_differences_1D...')

        u = np.linspace(0, 10, 11)
        phi = np.linspace(-5, 5, 11)

        ce = extrapolator.CE(u, phi)
        upwind_u_diff = ce.compute_upwind_differences()

        print('\tTest complete.')


    def test_upwind_differences_2D(self):
        print('\n\n\tRunning test_upwind_differences_2D...')

        x = np.linspace(0, 10, 11)
        y = np.linspace(0, 10, 11)

        xv, yv = np.meshgrid(x, y)

        u = xv + yv
        phi = u - 5

        ce = extrapolator.CE(u, phi)
        upwind_u_diff = ce.compute_upwind_differences()

        print('\tTest complete.')

    def test_solve_circle_const(self):
        print('\n\n\tRunning test_solve_circle_const...')

        mask = np.zeros((100, 100), dtype=np.bool)
        rr, cc = circle(50, 50, 25)
        mask[rr, cc] = True
        mask_inverted = np.logical_not(mask)

        inside_distance = distance_transform_edt(mask)
        outside_distance = distance_transform_edt(mask_inverted)
        circle_level_set = outside_distance - inside_distance

        u = np.zeros(mask.shape)
        u[mask] = 1.0
        u[45:55, 45:55] = -1.0

        ce = extrapolator.CE(u.copy(), circle_level_set)
        u_n = ce.solve()

        print('\tMean(u_n) =', np.mean(u_n[np.nonzero(mask_inverted)]))
        assert np.abs(np.mean(u_n[np.nonzero(mask_inverted)]) - 1.0) < 1e-8
        print('\tTest complete.')

    def test_solve_circle(self):
        print('\n\n\tRunning test_solve_circle...')

        mask = np.zeros((100, 100), dtype=np.bool)
        rr, cc = circle(50, 50, 25)
        mask[rr, cc] = True
        mask_inverted = np.logical_not(mask)

        inside_distance = distance_transform_edt(mask)
        outside_distance = distance_transform_edt(mask_inverted)
        circle_level_set = outside_distance - inside_distance

        u = np.zeros(mask.shape)
        u = np.indices(u.shape)[1]
        u[mask_inverted] = 0.0

        ce = extrapolator.CE(u.copy(), circle_level_set)
        u_n = ce.solve()

        fig, axs = plt.subplots(3, 1, figsize=(5, 15))

        im = axs[0].imshow(circle_level_set)
        c = axs[0].contour(circle_level_set, colors='white')
        axs[0].clabel(c, inline=1, fontsize=10)
        axs[0].set_title('Level set field')
        fig.colorbar(im, ax=axs[0])

        im = axs[1].imshow(u)
        c = axs[1].contour(u, colors='white')
        axs[1].clabel(c, inline=1, fontsize=10)
        axs[1].set_title('Initial u')
        fig.colorbar(im, ax=axs[1])

        im = axs[2].imshow(u_n)
        c = axs[2].contour(u_n, colors='white')
        axs[2].clabel(c, inline=1, fontsize=10)
        axs[2].set_title('Extrapolated u')
        fig.colorbar(im, ax=axs[2])

        outname = './tests/test_solve_circle_output.png'
        plt.savefig(outname)

        print('\tOutput written to', outname)
        print('\tTest complete.')
