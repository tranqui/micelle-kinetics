#!/usr/bin/env python3

import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import basico as bs

try: from micellethermo import MicelleDistribution
except: from .micellethermo import MicelleDistribution


class AnianssonWall:
    def __init__(self, n, max_n_exchanged=1):
        """Model for micelle formation by stepwise association/dissociation of
        monomer.

        Args:
            n: maximum value of n considered.
            max_n_exchanged: maximum sized aggregate that can be exchanged.
                             This is =1 for strict Aniansson-Wall model of
                               single monomer exchange, but can be set to
                               larger to allow for generalised kinetics
                               (e.g. fission/fusion events).
        """

        self.n_max = n
        assert max_n_exchanged >= 1
        self.max_n_exchanged = max_n_exchanged
        self.calculate_stoichiometry()

    def steady_state(self, n):
        raise NotImplementedError

    @property
    def n(self):
        """Size of all aggregates considered in problem."""
        return 1 + np.arange(self.n_max)

    @property
    def num_reactions(self):
        r"""Number of distinct chemical pathways.

        This counts the number of different forward *and* backward reactions
        of the form:
            $\ce{A_i + A_j <=> A_{i+j}$
        where i and j must be less than or equal to max_n_exchanged, and
        i + j must be less than n_max. This gives
            $2 \sum_{k=1}^m (n - 2k + 1) = 2m (n - m)$
        where the 2 in front accounts for forward and backward, m is
        max_n_exchanged and n is n_max.
        """

        return 2*self.max_n_exchanged * (self.n_max - self.max_n_exchanged)

    @property
    def S(self):
        """Alias for stoichiometry matrix."""
        return self.stoichiometry

    def velocity_gradient(self, c):
        kf, kb = self.rate_coeffs
        V = np.zeros((self.num_reactions, self.n_max))
        i = 2*np.arange(self.num_reactions//2)
        V[i, self.i1] += kf * c[self.i2]
        V[i, self.i2] += kf * c[self.i1]
        V[i+1, self.iprod] = kb
        return V

    def rhs(self, c):
        rates = np.zeros(self.num_reactions)
        kf, kb = self.rate_coeffs
        rates[::2] = kf * c[self.i1] * c[self.i2]
        rates[1::2] = kb * c[self.iprod]
        return self.S.T.dot(rates)

    def jacobian_rhs(self, c):
        return self.S.T.dot(self.velocity_gradient(c))

    def characteristic_rates(self, c):
        J = self.jacobian_rhs(c)
        evals = np.linalg.eigvals(J)
        evals = evals[np.abs(evals) > 1e-8]
        return np.sort(evals)

    def calculate_stoichiometry(self):
        S = np.zeros((self.num_reactions, self.n_max), dtype=int)

        # Indices of reactants and products
        i1, i2 = np.triu_indices(self.n_max)
        iprod = i1 + i2 + 1
        select = np.logical_and(i1 < self.max_n_exchanged, iprod < self.n_max)
        i1, i2, iprod = i1[select], i2[select], iprod[select]
        assert 2*len(i1) == self.num_reactions
        self.i1, self.i2, self.iprod = i1, i2, iprod

        # Forward reactions Ai1 + Ai2 --> A(i1+i2) indexed by even indices
        for i in range(self.num_reactions//2):
            S[2*i, self.i1[i]] -= 1
            S[2*i, self.i2[i]] -= 1
            S[2*i, self.iprod[i]] += 1

        # Backwards reactions (indexed by odd indices) are simply the reverse
        S[1::2] = -S[::2]

        self.stoichiometry = S

    def rate_ratio(self, k):
        """Ratio kf / kb for kth chemical reaction which has the generic form
            A_{i1+1} + A_{i2+1} <-> A_{i1+i2+2}
        where the specific aggregate numbers were calculated previously
        in calculate_stoichiometry.

        Args:
            k: index of specific chemical reaction.
        """

        n1, n2 = 1 + self.i1[k], 1 + self.i2[k]
        return self.steady_state(1+self.iprod[k]) / \
              (self.steady_state(n1) * self.steady_state(n2))

    def excess_free_energy_change(self, n):
        """Implied excess free energy change between A_n and A_{n+1}."""
        return self.steady_state(n+1) / self.steady_state(n)

    @property
    def n_local_maximum(self):
        dG = self.excess_free_energy_change(self.n[:-1])
        return 1 + np.flatnonzero(dG > 1)[0]

    def calculate_rate_coeffs(self):
        """Calculate coefficients of rates for single-monomer exchange
        (association and dissociation) events. This assumes diffusion-limited
        reaction rates of Smoluchowski (1917) for the forward rates, and the
        backwards rates are calculated from the known steady-state profile."""

        # Assume diffuion-limited rates of Smoluchowski (1917):
        # Note: we use time units which set dimer formation to unity.
        radii = self.n**(1/3)
        R1 = radii[self.i1]
        R2 = radii[self.i2]
        D = 1/R1 + 1/R2 # mutual diffusion of products
        kf = (R1 + R2) * D
        kf /= kf[0]
        kb = kf / self.rate_ratio(np.arange(self.num_reactions//2))

        self.rate_coeffs = kf, kb

    def integrate(self, c0, t, dt=1e-2, t_eval=None):
        """Integrate deterministic chemical kinetic equations via scipy.
        
        Args:
            c0: initial condition (concentration vector).
            t: total time to integrate for.
            dt: timestep for integration.
            t_eval: if set will evaluate concentrations only at these times.
        Returns:
            The concentration at each time.
        """
        jac = lambda t, c: self.jacobian_rhs(c)
        rates = lambda t, c: self.rhs(c)
        sol = solve_ivp(rates, [0, t], c0, max_step=dt, jac=jac, rtol=1e-4,
                        t_eval=t_eval)

        if t_eval is None: return sol.y[:,-1]
        else: return sol.y

    def load_copasi_model(self, N, density=1):
        """Load model into COPASI via basico for efficient integration of
        chemical equations (deterministic or stochastic).

        The initial condition is hard-coded in the model. Currently we assume
        that the system is begun with an all-monomer solution.

        Args:
            N: number of particles in the simulation. This is needed for
                stochastic simulations but doesn't have any effect for
                deterministic.
            density: sets the total amount of surfactant (needed for the
                initial condition).
        Returns:
            The created basico model object.
        """
        V = N / density

        model = bs.new_model()
        bs.set_model_unit(quantity_unit='#')
        bs.add_compartment('system', V)

        for i in range(self.n_max):
            species = f'A{i+1}'
            initial_conc = density if species == 'A1' else 0
            bs.add_species(species, initial_concentration=initial_conc)

        bs.add_reaction('R1f', '2 * A1 -> A2') # special cases
        bs.add_reaction('R1b', 'A2 -> 2 * A1') 

        for i in range(1, self.num_reactions//2): # all the other cases
            bs.add_reaction(f'R{i+1}f', f'A1 + A{i+1} -> A{i+2}')
            bs.add_reaction(f'R{i+1}b', f'A{i+2} -> A1 + A{i+1}')

        kf, kb = self.rate_coeffs
        for i in range(self.num_reactions//2):
            bs.set_reaction_parameters(f'(R{i+1}f).k1', value=kf[i])
            bs.set_reaction_parameters(f'(R{i+1}b).k1', value=kb[i])

        nvars = [f'A{i+1}.ParticleNumber' for i in range(self.n_max)]
        cvars = [f'[A{i+1}]' for i in range(self.n_max)]
        nvars_weighted = [f'{i+1}*{var}' for i, var in enumerate(nvars)]

        nfree = ' + '.join(nvars_weighted[:self.n_local_maximum])
        nmicelle = ' + '.join(nvars[self.n_local_maximum:])
        pfree = f'Values[nfree] / {N}'
        cfree = f'Values[nfree] / {V}'
        cmicelle = f'Values[nmicelle] / {V}'

        n_in_micelle = f'{N} - Values[nfree]'
        navg = f'({n_in_micelle}) / Values[nmicelle]'

        bs.add_parameter('nfree', type='assignment', expression=nfree)
        bs.add_parameter('nmicelle', type='assignment', expression=nmicelle)
        bs.add_parameter('pfree', type='assignment', expression=pfree)
        bs.add_parameter('cfree', type='assignment', expression=cfree)
        bs.add_parameter('cmicelle', type='assignment', expression=cmicelle)
        bs.add_parameter('navg', type='assignment', expression=navg)

        return model

    def copasi_integrate(self, t, step=1e-2, nruns=1, N=100,
                         method='lsoda', t_eval=None):
        """Integrate chemical kinetic equations via COPASI.
        
        Args:
            t: total time to integrate for.
            step: timestep.
            N: number of particles. This does not have any effect for
                deterministic simulations.
            method: 'lsoda' for deterministic integration and 'stochastic'
                    for stochastic.
            t_eval: if given the system state will be evaluated at these times,
                    otherwise it will be evaluated every timestep.
        Returns:
            The system state paramaters at each evaluated time.
        """

        sim = self.load_copasi_model(N=N)

        params = bs.get_parameters().index
        cols = {'Time': 'time'} | {f'Values[{x}]': x for x in params}

        data = []
        for i in range(nruns):
            kwargs = dict(duration=t, stepsize=step, method=method)
            if t_eval is not None: kwargs['values'] = t_eval
            df = bs.run_time_course_with_output(cols.keys(), **kwargs)
            df.rename(columns=cols, inplace=True)
            data += [df]

        if nruns == 1: data = data[0]

        bs.remove_datamodel(sim)

        return data


class AnianssonWallCartoon(AnianssonWall):
    def __init__(self, n,
                 lin_decay=-np.log(2e-3)/5,
                 gauss_mean=30., gauss_sigma=4., gauss_weight=1.5,
                 **kwargs):
        """
        Assumes equilibrium profile with an exponentially decaying distribution
        of pre-micellar aggregates and a Gaussian mode for true micelles.

        Args:
            n: maximum value of n considered.
            gauss_mean: mean of Gaussian peak for true micelles
            gauss_sigma: stddev of Gaussian peak for true micelles
            gauss_weight: weight of Gaussian (i.e. multiplicative prefactor)
            kwargs: any other arguments to the integrator.
        """

        super().__init__(n, **kwargs)
 
        self.lin_decay = lin_decay
        self.gauss_mean = gauss_mean
        self.gauss_sigma = gauss_sigma
        self.gauss_weight = gauss_weight * np.sqrt(2 * np.pi * gauss_sigma**2)

        self.calculate_rate_coeffs()

    def steady_state(self, n):
        premicellar = np.exp(-self.lin_decay*(n-1))
        gaussian = np.exp(-(n-self.gauss_mean)**2/(2*self.gauss_sigma**2)) / \
                   np.sqrt(2 * np.pi * self.gauss_sigma**2)
        return premicellar + self.gauss_weight * gaussian


class AnianssonWallMaibaum(AnianssonWall):
    """More realistic model using the thermodynamic arguments from:
        Maibaum et al., J. Phys. Chem. B 108(21), 6778-7681 (2004).
    """

    def __init__(self, size_ratio=None,
                 target_p_free=None,
                 target_n_micelle=None,
                 density=None, free_monomer_concentration=None,
                 kinetic_length='inverse',
                 g0=20,
                 distribution=None,
                 **kwargs):
        r"""Construct thermodynamic potential governing equilibrium distribution.

        Either the density or the free monomer concentration can be set to fix
        the absolute magnitude of the concentration profile, but both cannot be
        fixed simultaneously. If neither are specified then the free monomer
        concentration will be set to one.

        Args:
            size_ratio: ratio girth / length that determines $h$.
            target_p_free: fraction of free surfactant in final distribution.
            target_n_micelle: the modal micelle size.
            density (optional): total density of monomer across whole system.
                Technically this is in units of [L]^{-3}, where [L] is some
                length unit. In Maibaum et al. (2004) they use the surfactant
                girth as the length unit.
            free_monomer_concentration (optional): concentration of free monomer
                (i.e. n=1 aggregates) that rescales the overall concentration
                profile. This also contributes the ideal part of
                the driving force for aggregation. This is in the same units
                as density.
            kinetic_length: length-scale $\Lambda$ used in ideal gas chemical
                potential $-\ln{c_1 \Lambda^3}$ where $c_1$ is the free monomer
                concentration. This only changes how the free energy is
                partitioned into excess and ideal parts.
                If 'inverse' this will be set to $c_1^{-3}$ thereby removing
                any effect of free monomer concentration on the free energy, so
                there is only an excess contribution.
            g0: initial guess for $g$ in the numerical optimisation. This may
                need to be tweaked to find the right solution.
            distribution: alternatively to the parameters above, the
                MicelleDistribution object can be specified directly.
            kwargs: any other arguments to the integrator.
        """

        if distribution is None:
            assert size_ratio is not None
            assert target_p_free is not None
            assert target_n_micelle is not None
            self.distribution = \
                MicelleDistribution.from_fit(size_ratio, target_p_free,
                                            target_n_micelle, density,
                                            free_monomer_concentration,
                                            kinetic_length,
                                            g0=g0)
        else:
            self.distribution = distribution

        super().__init__(self.distribution.n_max, **kwargs)
        self.calculate_rate_coeffs()

    def steady_state(self, n):
        return self.distribution.steady_state(n)

    @property
    def density(self):
        return self.distribution.density

    def load_copasi_model(self, N):
        return super().load_copasi_model(N, density=self.density)


def test_steady_state(max_n_exchanged=1, n=50):
    """Test flux (i.e. the rhs dc/dt) vanishes in steady-state."""
    model = AnianssonWallCartoon(n, max_n_exchanged=max_n_exchanged)
    assert np.allclose(model.rhs(model.steady_state(model.n)), 0)


def test_jacobian(max_n_exchanged=1, n=50):
    """Test analytic jacobian is gradient of rhs."""
    model = AnianssonWallCartoon(n, max_n_exchanged=max_n_exchanged)
    c = np.random.random(n)
    from scipy.optimize import approx_fprime
    exact = approx_fprime(c, model.rhs)
    assert np.allclose(model.jacobian_rhs(c), exact, atol=1e-6)


if __name__ == '__main__':
    # Run unit tests:
    for m in range(1, 10):
        test_steady_state(m)
        test_jacobian(m)
