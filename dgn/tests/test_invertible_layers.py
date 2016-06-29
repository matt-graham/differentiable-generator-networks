import dgn.invertible_layers
import numpy as np
import theano.tensor as tt
import scipy.linalg as la
import numpy.linalg as np_la

SEED = 1234
N_DIM = 100
N_BATCH = 1000
A_TOL = 1e-5
R_TOL = 1e-5
FD_STEP = 1e-4
FD_TOL = 1e-6
D_TYPE = np.float32


class BaseLayerTestSuite(object):

    @classmethod
    def setup_class(cls):
        cls.prng = np.random.RandomState(SEED)
        cls.x_batch = cls.prng.normal(
            size=(N_BATCH, N_DIM)).astype(D_TYPE)
        cls.x_point = cls.prng.normal(size=N_DIM).astype(D_TYPE)
        cls.y_batch = cls.prng.normal(
            size=(N_BATCH, N_DIM)).astype(D_TYPE)
        cls.y_point = cls.prng.normal(size=N_DIM).astype(D_TYPE)

    def forward_map(self, x):
        raise NotImplementedError()

    def inverse_map(self, y):
        raise NotImplementedError()

    def forward_jacobian_log_det(self, x):
        raise NotImplementedError()

    def param_log_prior(self):
        raise NotImplementedError()

    def test_forward_map_batch(self):
        y_batch_layer = self.layer.forward_map_batch(self.x_batch)
        y_batch_test = self.forward_map(self.x_batch)
        assert np.allclose(y_batch_layer, y_batch_test,
                           rtol=R_TOL, atol=A_TOL), (
            'Max. abs. difference between layer forward output and test: {0}'
            .format(np.max(np.abs(y_batch_layer - y_batch_test))))

    def test_forward_map_point(self):
        y_point_layer = self.layer.forward_map_point(self.x_point)
        y_point_test = self.forward_map(self.x_point)
        assert np.allclose(y_point_layer, y_point_test,
                           rtol=R_TOL, atol=A_TOL), (
            'Max. abs. difference between layer forward output and test: {0}'
            .format(np.max(np.abs(y_point_layer - y_point_test))))

    def test_inverse_map_batch(self):
        x_batch_layer = self.layer.inverse_map_batch(self.y_batch)
        x_batch_test = self.inverse_map(self.y_batch)
        assert np.allclose(x_batch_layer, x_batch_test,
                           rtol=R_TOL, atol=A_TOL), (
            'Max. abs. difference between layer inverse output and test: {0}'
            .format(np.max(np.abs(x_batch_layer - x_batch_test))))

    def test_inverse_map_point(self):
        x_point_layer = self.layer.inverse_map_point(self.y_point)
        x_point_test = self.inverse_map(self.y_point)
        assert np.allclose(x_point_layer, x_point_test,
                           rtol=R_TOL, atol=A_TOL), (
            'Max. abs. difference between layer inverse output and test: {0}'
            .format(np.max(np.abs(x_point_layer - x_point_test))))

    def test_forward_inverse_batch_consistent(self):
        y_batch_fi = self.layer.forward_map_batch(
            self.layer.inverse_map_batch(self.y_batch))
        assert np.allclose(y_batch_fi, self.y_batch,
                           rtol=R_TOL, atol=A_TOL), (
            'Maximum absolute difference between layer '
            'forward-inverse output and test: {0}'
            .format(np.max(np.abs(y_batch_fi - y_batch))))

    def test_forward_inverse_point_consistent(self):
        y_point_fi = self.layer.forward_map_point(
            self.layer.inverse_map_point(self.y_point))
        assert np.allclose(y_point_fi, self.y_point,
                           rtol=R_TOL, atol=A_TOL), (
            'Maximum absolute difference between layer '
            'forward-inverse output and test: {0}'
            .format(np.max(np.abs(y_point_fi - y_point))))

    def test_inverse_forward_batch_consistent(self):
        x_batch_if = self.layer.inverse_map_batch(
            self.layer.forward_map_batch(self.x_batch))
        assert np.allclose(x_batch_if, self.x_batch,
                           rtol=R_TOL, atol=A_TOL), (
            'Maximum absolute difference between layer '
            'inverse-forward output and test: {0}'
            .format(np.max(np.abs(x_batch_if - x_batch))))

    def test_inverse_forward_point_consistent(self):
        x_point_if = self.layer.inverse_map_point(
            self.layer.forward_map_point(self.x_point))
        assert np.allclose(x_point_if, self.x_point,
                           rtol=R_TOL, atol=A_TOL), (
            'Maximum absolute difference between layer '
            'inverse-forward output and test: {0}'
            .format(np.max(np.abs(x_point_if - x_point))))

    def test_log_det_forward_map_jacobian_batch(self):
        log_det_jacob_layer = (
            self.layer.forward_jacobian_log_det_batch(self.x_batch))
        log_det_jacob_test = self.forward_jacobian_log_det(self.x_batch)
        assert np.allclose(log_det_jacob_layer, log_det_jacob_test,
                           rtol=R_TOL, atol=A_TOL), (
            'Maximum absolute difference between layer '
            'forward jacobian log determinant and test: {0}'
            .format(np.max(np.abs(log_det_jacob_layer - log_det_jacob_test))))

    def test_log_det_forward_map_jacobian_point(self):
        log_det_jacob_layer = (
            self.layer.forward_jacobian_log_det_point(self.x_point))
        log_det_jacob_test = self.forward_jacobian_log_det(self.x_point)
        assert np.allclose(log_det_jacob_layer, log_det_jacob_test,
                           rtol=R_TOL, atol=A_TOL), (
            'Maximum absolute difference between layer '
            'forward jacobian log determinant and test: {0}'
            .format(np.max(np.abs(log_det_jacob_layer - log_det_jacob_test))))

    def test_log_det_forward_map_jacobian_fd(self):
        jacob_fd = np.zeros((N_DIM, N_DIM))
        for i in range(N_DIM):
            #h = np.zeros(N_DIM).astype(D_TYPE)
            #h[i] = FD_STEP
            #y_pl = self.layer.forward_map_point(self.x_point + h)
            #y_mn = self.layer.forward_map_point(self.x_point - h)
            #jacob_fd[i] = (y_pl - y_mn) / (2 * FD_STEP)
            h = np.zeros(N_DIM).astype(np.complex128)
            h[i] = FD_STEP * 1j
            jacob_fd[i] = self.forward_map(self.x_point + h).imag / FD_STEP
        _, log_det_jacob_fd = np_la.slogdet(jacob_fd)
        log_det_jacob_layer = (
            self.layer.forward_jacobian_log_det_point(self.x_point))
        assert np.abs(log_det_jacob_layer - log_det_jacob_fd) < FD_TOL, (
            'Maximum absolute difference between layer '
            'forward jacobian log det. and finite difference approx.: {0}'
            .format(np.max(np.abs(log_det_jacob_layer - log_det_jacob_fd))))

    def test_param_log_prior(self):
        param_log_prior_layer = self.layer.param_log_prior().eval()
        param_log_prior_test = self.param_log_prior()
        assert np.allclose(param_log_prior_layer, param_log_prior_test,
                           rtol=R_TOL, atol=A_TOL), (
            'Maximum absolute difference between layer '
            'parameter log prior and test: {0}'
            .format(np.max(np.abs(param_log_prior_layer -
                                  param_log_prior_test))))


class TestAffineLayer(BaseLayerTestSuite):

    @classmethod
    def setup_class(cls):
        super(TestAffineLayer, cls).setup_class()
        cls.W = (cls.prng.uniform(-0.05, 0.05, size=(N_DIM, N_DIM)) +
                 np.eye(N_DIM)).astype(D_TYPE)
        cls.b = (cls.prng.normal(size=N_DIM)).astype(D_TYPE)
        cls.W_prec = 0.1
        cls.b_prec = 0.1
        cls.W_mean = np.eye(N_DIM).astype(D_TYPE)
        cls.b_mean = np.zeros(N_DIM).astype(D_TYPE)
        cls.layer = invertible_layers.AffineLayer(
            weights_init=cls.W,
            biases_init=cls.b,
            weights_prec=cls.W_prec,
            biases_prec=cls.b_prec,
            weights_mean=cls.W_mean,
            biases_mean=cls.b_mean)
        cls.layer.compile_theano_functions()

    def forward_map(self, x):
        return x.dot(self.W.T) + self.b

    def inverse_map(self, y):
        return la.solve(self.W, (y - self.b).T).T

    def forward_jacobian_log_det(self, x):
        _, jld = np_la.slogdet(self.W)
        if x.ndim == 1:
            return jld
        elif x.ndim == 2:
            return x.shape[0] * jld
        else:
            raise ValueError('x must be one or two dimensional.')

    def param_log_prior(self):
        return -(0.5 * self.W_prec *
                ((self.W - self.W_mean)**2).sum() +
                 0.5 * self.b_prec *
                ((self.b - self.b_mean)**2).sum())


class TestLowerTriangularAffineLayer(BaseLayerTestSuite):

    @classmethod
    def setup_class(cls):
        super(TestLowerTriangularAffineLayer, cls).setup_class()
        cls.W = np.tril(cls.prng.uniform(-0.05, 0.05, size=(N_DIM, N_DIM)) +
                        np.eye(N_DIM)).astype(D_TYPE)
        cls.b = (cls.prng.normal(size=N_DIM)).astype(D_TYPE)
        cls.W_prec = 0.1
        cls.b_prec = 0.1
        cls.W_mean = np.eye(N_DIM).astype(D_TYPE)
        cls.b_mean = np.zeros(N_DIM).astype(D_TYPE)
        cls.layer = invertible_layers.TriangularAffineLayer(
            weights_init=cls.W,
            biases_init=cls.b,
            lower=True,
            weights_prec=cls.W_prec,
            biases_prec=cls.b_prec,
            weights_mean=cls.W_mean,
            biases_mean=cls.b_mean)
        cls.layer.compile_theano_functions()

    def forward_map(self, x):
        return x.dot(self.W.T) + self.b

    def inverse_map(self, y):
        return la.solve_triangular(self.W, (y - self.b).T, lower=True).T

    def forward_jacobian_log_det(self, x):
        jld = np.log(np.abs(self.W.diagonal())).sum()
        if x.ndim == 1:
            return jld
        elif x.ndim == 2:
            return x.shape[0] * jld
        else:
            raise ValueError('x must be one or two dimensional.')

    def param_log_prior(self):
        return -(0.5 * self.W_prec *
                ((self.W - self.W_mean)**2).sum() +
                 0.5 * self.b_prec *
                ((self.b - self.b_mean)**2).sum())


class TestUpperTriangularAffineLayer(BaseLayerTestSuite):

    @classmethod
    def setup_class(cls):
        super(TestUpperTriangularAffineLayer, cls).setup_class()
        cls.W = np.triu(cls.prng.uniform(-0.05, 0.05, size=(N_DIM, N_DIM)) +
                        np.eye(N_DIM)).astype(D_TYPE)
        cls.b = (cls.prng.normal(size=N_DIM)).astype(D_TYPE)
        cls.W_prec = 0.1
        cls.b_prec = 0.1
        cls.W_mean = np.eye(N_DIM).astype(D_TYPE)
        cls.b_mean = np.zeros(N_DIM).astype(D_TYPE)
        cls.layer = invertible_layers.TriangularAffineLayer(
            weights_init=cls.W,
            biases_init=cls.b,
            lower=False,
            weights_prec=cls.W_prec,
            biases_prec=cls.b_prec,
            weights_mean=cls.W_mean,
            biases_mean=cls.b_mean)
        cls.layer.compile_theano_functions()

    def forward_map(self, x):
        return x.dot(self.W.T) + self.b

    def inverse_map(self, y):
        return la.solve_triangular(self.W, (y - self.b).T, lower=False).T

    def forward_jacobian_log_det(self, x):
        jld = np.log(np.abs(self.W.diagonal())).sum()
        if x.ndim == 1:
            return jld
        elif x.ndim == 2:
            return x.shape[0] * jld
        else:
            raise ValueError('x must be one or two dimensional.')

    def param_log_prior(self):
        return -(0.5 * self.W_prec *
                ((self.W - self.W_mean)**2).sum() +
                 0.5 * self.b_prec *
                ((self.b - self.b_mean)**2).sum())


class TestDiagonalAffineLayer(TestAffineLayer):

    @classmethod
    def setup_class(cls):
        super(TestAffineLayer, cls).setup_class()
        cls.d = (1. + cls.prng.uniform(-0.05, 0.05, size=N_DIM)).astype(D_TYPE)
        cls.W = np.diag(cls.d)
        cls.b = (cls.prng.normal(size=N_DIM)).astype(D_TYPE)
        cls.d_prec = 0.1
        cls.b_prec = 0.1
        cls.d_mean = np.ones(N_DIM).astype(D_TYPE)
        cls.b_mean = np.zeros(N_DIM).astype(D_TYPE)
        cls.layer = invertible_layers.DiagonalAffineLayer(
            diag_weights_init=cls.d,
            biases_init=cls.b,
            diag_weights_prec=cls.d_prec,
            biases_prec=cls.b_prec,
            diag_weights_mean=cls.d_mean,
            biases_mean=cls.b_mean)
        cls.layer.compile_theano_functions()

    def param_log_prior(self):
        return -(0.5 * self.d_prec *
                ((self.d - self.d_mean)**2).sum() +
                 0.5 * self.b_prec *
                ((self.b - self.b_mean)**2).sum())


class TestDiagPlusRank1AffineLayer(TestAffineLayer):

    @classmethod
    def setup_class(cls):
        super(TestAffineLayer, cls).setup_class()
        cls.d = (1. + cls.prng.uniform(-0.05, 0.05, size=N_DIM)).astype(D_TYPE)

        cls.u, cls.v = cls.prng.uniform(
            -0.05, 0.05, size=(2, N_DIM)).astype(D_TYPE)
        cls.W = np.diag(cls.d) + np.outer(cls.u, cls.v)
        cls.b = (cls.prng.normal(size=N_DIM)).astype(D_TYPE)
        cls.d_prec = 0.1
        cls.u_prec = 0.1
        cls.v_prec = 0.1
        cls.b_prec = 0.1
        cls.d_mean = np.ones(N_DIM).astype(D_TYPE)
        cls.u_mean, cls.v_mean, cls.b_mean = np.zeros(
            (3, N_DIM)).astype(D_TYPE)
        cls.layer = invertible_layers.DiagPlusRank1AffineLayer(
            diag_weights_init=cls.d,
            u_vect_init=cls.u,
            v_vect_init=cls.v,
            biases_init=cls.b,
            diag_weights_prec=cls.d_prec,
            u_vect_prec=cls.u_prec,
            v_vect_prec=cls.v_prec,
            biases_prec=cls.b_prec,
            diag_weights_mean=cls.d_mean,
            u_vect_mean=cls.u_mean,
            v_vect_mean=cls.v_mean,
            biases_mean=cls.b_mean)
        cls.layer.compile_theano_functions()

    def param_log_prior(self):
        return -(0.5 * self.d_prec *
                ((self.d - self.d_mean)**2).sum() +
                 0.5 * self.u_prec *
                ((self.u - self.u_mean)**2).sum() +
                 0.5 * self.v_prec *
                ((self.v - self.v_mean)**2).sum() +
                 0.5 * self.b_prec *
                ((self.b - self.b_mean)**2).sum())


class TestElementwiseLayer(BaseLayerTestSuite):

    @classmethod
    def setup_class(cls):
        super(TestElementwiseLayer, cls).setup_class()
        cls.forward_func = np.sinh
        cls.forward_deriv = np.cosh
        cls.inverse_func = np.arcsinh
        cls.layer = invertible_layers.ElementwiseLayer(
            forward_func=tt.sinh,
            inverse_func=tt.arcsinh
        )
        cls.layer.compile_theano_functions()

    def forward_map(self, x):
        return self.forward_func(x)

    def inverse_map(self, y):
        return self.inverse_func(y)

    def forward_jacobian_log_det(self, x):
        return np.log(self.forward_deriv(x)).sum()

    def param_log_prior(self):
        return 0.


class TestPermuteDimensionsLayer(BaseLayerTestSuite):

    @classmethod
    def setup_class(cls):
        super(TestPermuteDimensionsLayer, cls).setup_class()
        cls.perm = cls.prng.permutation(N_DIM)
        cls.inv_perm = np.argsort(cls.perm)
        cls.layer = invertible_layers.PermuteDimensionsLayer(perm=cls.perm)
        cls.layer.compile_theano_functions()

    def forward_map(self, x):
        if x.ndim == 1:
            return x[self.perm]
        elif x.ndim == 2:
            return x[:, self.perm]
        else:
            raise ValueError('x must be one or two dimensional.')

    def inverse_map(self, y):
        if y.ndim == 1:
            return y[self.inv_perm]
        elif y.ndim == 2:
            return y[:, self.inv_perm]
        else:
            raise ValueError('y must be one or two dimensional.')

    def forward_jacobian_log_det(self, x):
        return 0.

    def param_log_prior(self):
        return 0.
