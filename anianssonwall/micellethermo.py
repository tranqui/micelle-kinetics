#!/usr/bin/env python3

import numpy as np
from scipy.optimize import minimize


def h(size_ratio):
    """Coefficient of stoichiometric term $h n^{5/3}$ in theory of
        Maibaum et al., J. Phys. Chem. B 108(21), 6778-7681 (2004).

    Args:
        size_ratio: surfactant girth / length (dimensionless).
    Returns:
        Coefficient $h$.
    """
    return (3/(4*np.pi))**(2/3) * (96/49) * (size_ratio)**(4/3)

def g(effective_tension, size_ratio):
    """Coefficient of surface term $g n^{2/3}$ in theory of
        Maibaum et al., J. Phys. Chem. B 108(21), 6778-7681 (2004).

    Args:
        effective_tension: surface tension * girth^2 (units of energy).
        size_ratio: surfactant girth / length (dimensionless).
    Returns:
        Coefficient $g$.
    """
    return (36*np.pi)**(1/3) * effective_tension / size_ratio**(2/3)

def effective_tension_from_g(g, size_ratio):
    """Invert the definition of g to pill out an effective tension."""
    return size_ratio**(2/3) * g / (36*np.pi)**(1/3)

def free_energy(n, h, g, e):
    r"""Free energy of forming an aggregate of size $n$ out of $n$ monomer in
    the theory of:
        Maibaum et al., J. Phys. Chem. B 108(21), 6778-7681 (2004).

    Args:
        n: aggregate size (# of molecules, $r \ge 1$)
        h: coefficient of stoichiometric penalty term ($h > 0$).
        g: coefficient of surface penalty term ($g > 0$).
        e: driving force for aggregation ($e > 0$).
    Returns:
        Free energy $\Delta G$.
    """

    bulk_term = -e * (n-1)
    surface_term = g * (n-1)**(2/3)
    stoichiometric_term = h * (n-1)**(5/3)
    return bulk_term + surface_term + stoichiometric_term

def h_crit(g, e):
    """Maximum value of $h$ to see micelle formation, i.e. where the free
    energy develops a local maximum.
    
    This condition is obtained from the discriminant of $G'(n)$.

    Args:
        g: coefficient of surface penalty term ($g > 0$).
        e: driving force for aggregation ($e > 0$).
    Returns;
        Maximum value of $h$ to see local maximum.
    """
    return e**3 / (5 * g**2)

def e_crit(h, g):
    """Minimum value of $e$ to see micelle formation, i.e. where the free
    energy develops a local maximum.
    
    This condition is obtained from the discriminant of $G'(n)$.

    Args:
        h: coefficient of stoichiometric penalty term ($h > 0$).
        g: coefficient of surface penalty term ($g > 0$).
    Returns:
        Minimum value of $e$ to develop a local maximum.
    """
    return (5 * h * g**2)**(1/3)

def local_maximum(h, g, e, eps=1e-8):
    """Position of local maximum in free energy or, equivalently, the least
    probable pre-micellar aggregate.

    Args:
        h: coefficient of stoichiometric penalty term ($h > 0$).
        g: coefficient of surface penalty term ($g > 0$).
        e: driving force for aggregation ($e > 0$).
    Returns:
        Position of local maximum (an aggregate size $n$).
    """

    assert e > 0
    assert g > 0
    if h < 0 or h > h_crit(g, e): return np.nan

    z = (e**3 - 5*g**2 * h)**0.5
    assert z >= 0 # must be true when h in valid bounds above

    Z = (-5*g*np.sqrt(h) + 5**0.5 * z * 1j)**(1/3)
    n_crit = 1 - 2*g / (5*h) + (
                3*(-1)**(2/3) * e**2 / (5**(4/3) * Z) - \
                3*(-1)**(1/3) * e * Z / (5**(5/3)) ) / h**1.5
    assert abs(n_crit.imag) < eps

    return n_crit.real

def local_minimum(h, g, e, eps=1e-8):
    """Position of local minima in free energy or, equivalently, the most
    probable micelle size.

    Note the number will not be an integer, so this represents a thermally
    averaged modal size. To obtain the true most probable micelle size one
    would need to round the result.

    Args:
        h: coefficient of stoichiometric penalty term ($h > 0$).
        g: coefficient of surface penalty term ($g > 0$).
        e: driving force for aggregation ($e > 0$).
    Returns:
        Position of local minima (an aggregate size $n$).
    """

    assert e > 0
    assert g > 0
    if h < 0 or h > h_crit(g, e): return np.nan

    z = (e**3 - 5*g**2 * h)**0.5
    assert z >= 0 # must be true when h in valid bounds above

    Z = (-5*g*h**0.5 + 5**0.5 * z * 1j)**(1/3)
    minima = 1 - 2*g / (5*h) + 3*e**2 / (5**(4/3) * h**1.5 * Z) + \
             3*e*Z / (5**(5/3) * h**1.5)
    assert abs(minima.imag) < eps

    return minima.real

def n_max(h, g, e, threshold_eps=1e-6):
    """Maximum value of $n$ to consider to capture main distribution of
    equilibrium micellar aggregates.

    The maximum aggregate size $n$ is determined by a numerical threshold
    of probability beyond the local maximum.

    Args:
        h: coefficient of stoichiometric penalty term ($h > 0$).
        g: coefficient of surface penalty term ($g > 0$).
        e: driving force for aggregation ($e > 0$).
        threshold_eps: for determining where probability is treated as zero.
    Returns:
        Maximum value of $n$ needed (int).
    """

    try: n_crit = round(local_minimum(h, g, e))
    except: n_crit = 1 # if there is no local maximum in free energy
    G_crit = free_energy(n_crit, h, g, e)

    n_trial = np.arange(n_crit, 1000*n_crit)
    dG_trial = free_energy(n_trial, h, g, e) - G_crit
    index = np.where(np.exp(-dG_trial) < threshold_eps)[0][0]
    return n_crit + index

def n_trial(h, g, e, threshold_eps=1e-6):
    """Provide a container of $n$ to capture the main distribution of
    aggregates in steady-state.

    The maximum aggregate size $n$ is determined by a numerical threshold
    of probability beyond the local maximum.

    Args:
        h: coefficient of stoichiometric penalty term ($h > 0$).
        g: coefficient of surface penalty term ($g > 0$).
        e: driving force for aggregation ($e > 0$).
        threshold_eps: for determining where probability is treated as zero.
    Returns:
        Array of trial values of $n$: [1, ..., n_max].
    """
    return 1 + np.arange(n_max(h, g, e, threshold_eps))

def p_free(h, g, e):
    """Fraction of molecules that are not contained in micelles.

    An aggregate is considered to be micellar if it is larger than the size
    at the local maximum in free energy.

    Args:
        h: coefficient of stoichiometric penalty term ($h > 0$).
        g: coefficient of surface penalty term ($g > 0$).
        e: driving force for aggregation ($e > 0$).
    Returns:
        Fraction of free surfactant, i.e. a number in [0, 1].
    """

    n = n_trial(h, g, e)
    n_crit = max(1, local_maximum(h, g, e))
    free = n <= n_crit

    G = free_energy(n, h, g, e)
    c = np.exp(-G)
    phi = np.sum(n * c)

    c_free = np.sum(n[free] * c[free])
    p = c_free / phi
    assert p >= 0 and p <= 1
    return p

def p_micelle(h, g, e):
    """Fraction of molecules that *are* contained in micelles.

    An aggregate is considered to be micellar if it is larger than the size
    at the local_maximum in free energy.

    Args:
        h: coefficient of stoichiometric penalty term ($h > 0$).
        g: coefficient of surface penalty term ($g > 0$).
        e: driving force for aggregation ($e > 0$).
    Returns:
        Fraction of surfactant in micelles, i.e. a number in [0, 1].
    """
    return 1 - p_free(h, g, e)

def optimise_parameters(size_ratio, target_p_free, target_n_micelle, g0=20):
    """Find optimal parameters $g$ and $e$ in equilibrium micelle distribution
    that have desired (i) p_free and (ii) size of local minimum that determines
    the modal micelle size.

    Args:
        size_ratio: ratio girth / length that determines $h$.
        target_p_free: fraction of free surfactant in final distribution.
        target_n_micelle: the modal micelle size.
        g0: initial guess for $g$ in the numerical optimisation. This may
            need to be tweaked to find the right solution.
    Returns:
        The parameters (h, g, e) giving the equilibrium distribution
        with the desired properties.
    """

    h0 = h(size_ratio)
    e0 = 2*e_crit(h0, g0)

    e_constraint_func1 = lambda g, e: e - e_crit(h0, g)
    e_constraint_func2 = lambda x: e_constraint_func1(x[0], x[1])
    e_constraint = {'type': 'ineq', 'fun': e_constraint_func2}

    objective_p = lambda g, e: (p_free(h0,g,e) / target_p_free - 1)**2
    objective_n = lambda g, e: (local_minimum(h0,g,e)/target_n_micelle - 1)**2
    objective = lambda x: objective_p(x[0],x[1]) + objective_n(x[0],x[1])

    options = dict()
    # options['disp'] = True  # uncomment to print convergence information

    with np.errstate(over='ignore'):
        g, e = minimize(objective, [g0, e0],
                        bounds=((0, None), (0, None)), # force g > 0 and e > 0
                        constraints=[e_constraint],
                        options=options).x

    return h0, g, e


class MicelleDistribution:
    """Model of stationary distribution of micelle aggregates for non-ionic
    surfactants using the thermodynamic arguments of:
        Maibaum et al., J. Phys. Chem. B 108(21), 6778-7681 (2004).

    This class provides an object-oriented interface to the routines above.    
    """

    @staticmethod
    def from_fit(size_ratio, target_p_free, target_n_micelle,
                 density=None, free_monomer_concentration=None,
                 kinetic_length='inverse', g0=20):
        r"""
        Construct micelle distribution that has desired (i) p_free and
        (ii) size of local minimum that determines the modal micelle size.

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
                (i.e. r=1 aggregates) that rescales the overall concentration
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
        Returns:
            The equilibrium distribution with the desired properties.
        """

        # Cannot set both simultaneously: must be one or the other.
        assert not ((density is not None) and
                    (free_monomer_concentration is not None))

        h, g, e = optimise_parameters(size_ratio, target_p_free,
                                      target_n_micelle, g0)

        if density is None and free_monomer_concentration is None:
            free_monomer_concentration = 1
        elif density is not None:
            # Work out density if free monomer concentration were unity
            n = n_trial(h, g, e)
            G = free_energy(n, h, g, e)
            c = np.exp(-G)
            unscaled_density = np.sum(n * c)
            # Rescale to achieve correct density.
            free_monomer_concentration = density / unscaled_density

        if kinetic_length == 'inverse':
            kinetic_length = free_monomer_concentration**(-1/3)

        mu_id = -np.log(free_monomer_concentration * kinetic_length**3)
        e_ex = e + mu_id
        return MicelleDistribution(h, g, e_ex, free_monomer_concentration,
                                   kinetic_length)

    def __init__(self, h, g, e_ex, free_monomer_concentration=1,
                 kinetic_length='inverse'):
        r"""Construct thermodynamic potential governing equilibrium distribution.

        Args:
            h: coefficient of stoichiometric penalty term ($h > 0$).
            g: coefficient of surface penalty term ($g > 0$).
            e_ex: excess chemical potential that acts as a driving force for
                  aggregation ($e_ex > 0$).
            free_monomer_concentration: monomer density in bulk that rescales
                the overall density. This also contributes the ideal part of
                the driving force for aggregation. Technically this is in units
                of [L]^{-3}, where [L] is some length unit. In
                Maibaum et al. (2004) they use the surfactant girth as the
                length unit.
            kinetic_length: length-scale $\Lambda$ used in ideal gas chemical
                potential $-\ln{c_1 \Lambda^3}$ where $c_1$ is the free monomer
                concentration. This only changes how the free energy is
                partitioned into excess and ideal parts.
                If 'inverse' this will be set to $c_1^{-3}$ thereby removing
                any effect of free monomer concentration on the free energy, so
                there is only an excess contribution.
        """
 
        self.h = h
        self.g = g
        self.e_ex = e_ex
        self.free_monomer_concentration = free_monomer_concentration

        if kinetic_length == 'inverse':
            self.kinetic_length = free_monomer_concentration**(-1/3)
        else: self.kinetic_length = kinetic_length

    @property
    def ideal_chemical_potential(self):
        r"""Ideal chemical potential of free surfactant:
            $\mu^\mathrm{id} = - \ln{c_1 \Lambda^3}$
        """
        return -np.log(self.free_monomer_concentration * self.kinetic_length**3)

    @property
    def e(self):
        """Total driving force for aggregation from excess and ideal parts."""
        return self.e_ex - self.ideal_chemical_potential

    @property
    def surface_tension_head(self):
        """Effective surface tension between hydrophobic head and solvent
        in units of kT / [L]^2. The unit of length [L] is the girth of the
        surfactant molecules.
        """
        return effective_tension_from_g(self.g)

    @property
    def local_maximum(self):
        """Position of free energy's local maximum or, equivalently, the least
        probable pre-micellar aggregate.

        Returns:
            Position of local maximum (an aggregate size $n$).
        """
        return local_maximum(self.h, self.g, self.e)

    @property
    def local_minimum(self):
        """Position of local minima in free energy or, equivalently, the most
        probable micelle size.

        Note the number will not be an integer, so this represents a thermally
        averaged modal size. To obtain the true most probable micelle size one
        would need to round the result.

        Returns:
            Position of local minima (an aggregate size $n$).
        """
        return local_minimum(self.h, self.g, self.e)

    @property
    def n_max(self):
        r"""Value of n above which the distribution is negligible.
        
        Technically the distribution applies for all n \to \infty, but
        at a large enough $n$ the probabilities effectively drop to zero so
        we can exclude them. The location of this threshold is determined by
        a numerical tolerance under the hood.

        Return:
            The maximum value of $n$ according to the implicit threshold.
        """
        return n_max(self.h, self.g, self.e)

    @property
    def n(self):
        r"""Values of $n$ which effectively contain the distribution for all
        practical purposes.
        
        Technically the distribution applies for all $n$ \to \infty, but
        at a large enough $n$ the probabilities effectively drop to zero so
        we can exclude them. The location of this threshold is determined by
        a numerical tolerance under the hood.

        Returns:
            A numpy array containing the sequence [1, ..., n_max].
        """
        return 1 + np.arange(self.n_max)

    def excess_free_energy(self, n):
        r"""Excess free energy change from moving $n$ free monomers into a
        micelle/aggregate of size $n$.

        We assume the form of Maibaum et al. (2004). However, we make one small
        change in that we use $(n-1)$ rather than $n$ so that this change goes
        to zero properly for monomers.

        Args:
            n: aggregate size.
        Returns:
            The change in the excess free energy $\Delta G$.
        """
        return free_energy(n, self.h, self.g, self.e_ex)

    def free_energy(self, n):
        r"""Total free energy change from moving $n$ free monomers into a
        micelle/aggregate of size $n$. This includes the ideal contribution, so
        it is essentially a potential of mean force.

        We assume the form of Maibaum et al. (2004). However, we make one small
        change in that we use $(n-1)$ rather than $n$ so that this change goes
        to zero properly for monomers.

        Args:
            n: aggregate size.
        Returns:
            The change in the total free energy $\Delta \Omega$.
        """
        return free_energy(n, self.h, self.g, self.e)

    def steady_state(self, n):
        r"""Steady-state concentration profile from the free energy.

        Args:
            n: aggregate size.
        Returns:
            The change in the total free energy $\Delta \Omega$.
        """
        return self.free_monomer_concentration * np.exp(-self.free_energy(n))

    @property
    def p_free(self):
        """Fraction of molecules that are not contained in micelles.

        An aggregate is considered to be micellar if it is larger than the size
        at the local maximum in free energy.

        Returns:
            Fraction of free surfactant, i.e. a number in [0, 1].
        """
        return p_free(self.h, self.g, self.e)

    @property
    def p_micelle(self):
        """Fraction of molecules that *are* contained in micelles.

        An aggregate is considered to be micellar if it is larger than the size
        at the local maximum in free energy.

        Returns:
            Fraction of surfactant in micelles, i.e. a number in [0, 1].
        """
        return p_micelle(self.h, self.g, self.e)

    @property
    def c_micelle(self):
        """Fraction of molecules that are not contained in micelles.

        An aggregate is considered to be micellar if it is larger than the size
        at the local maximum in free energy.

        Args:
            h: coefficient of stoichiometric penalty term ($h > 0$).
            g: coefficient of surface penalty term ($g > 0$).
            e: driving force for aggregation ($e > 0$).
        Returns:
            Fraction of free surfactant, i.e. a number in [0, 1].
        """

        n = n_trial(h, g, e)
        n_crit = max(1, local_maximum(h, g, e))
        free = n <= n_crit

        G = free_energy(n, h, g, e)
        c = np.exp(-G)
        phi = np.sum(n * c)

        c_free = np.sum(n[free] * c[free])
        p = c_free / phi
        assert p >= 0 and p <= 1
        return p

    @property
    def density(self):
        return np.sum(self.n * self.steady_state(self.n))
