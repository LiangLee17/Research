// ---------------------------------------------------------------------
// $Id: constraints.h 30784 2013-09-18 07:07:32Z kronbichler $
//
// Copyright (C) 2010 - 2013 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------


/**
 * @defgroup constraints Constraints on degrees of freedom
 * @ingroup dofs
 *
 * This module deals with constraints on degrees of
 * freedom. The central class to deal with constraints is the ConstraintMatrix
 * class.
 *
 * Constraints typically come from several sources, for example:
 * - If you have Dirichlet-type boundary conditions, $u|_{\partial\Omega}=g$,
 *   one usually enforces
 *   them by requiring that degrees of freedom on the boundary have
 *   particular values, for example $x_{12}=42$ if the boundary condition
 *   $g(\mathbf x)$ requires that the finite element solution $u(\mathbf x)$
 *   at the location of degree
 *   of freedom 12 has the value 42. Such constraints are generated by
 *   those versions of the VectorTools::interpolate_boundary_values
 *   function that take a ConstraintMatrix argument (though there are
 *   also other ways of dealing with Dirichlet conditions, using
 *   MatrixTools::apply_boundary_values, see for example step-3 and step-4).
 * - If you have boundary conditions that set a certain part of the
 *   solution's value, for example no normal flux, $\mathbf n \cdot
 *   \mathbf u=0$ (as happens in flow problems and is handled by the
 *   VectorTools::compute_no_normal_flux_constraints function) or
 *   prescribed tangential components, $\vec{n}\times\vec{u}=
 *   \vec{n}\times\vec{f}$ (as happens in electromagnetic problems and
 *   is handled by the VectorTools::project_boundary_values_curl_conforming
 *   function). For the former case, imagine for example that we are at
 *   at vertex where the normal vector has the form $\frac 1{\sqrt{14}}
 *   (1,2,3)^T$ and that the $x$-, $y$- and $z$-components of the flow
 *   field at this vertex are associated with degrees of freedom 12, 28,
 *   and 40. Then the no-normal-flux condition means that we need to have
 *   the condition $\frac 1{\sqrt{14}} (x_{12}+2x_{28}+3x_{40})=0$.
 *   The prescribed tangential component leads to similar constraints
 *   though there is often something on the right hand side.
 * - If you have hanging node constraints, for example in a mesh like this:
 *        @image html hanging_nodes.png ""
 *   Let's assume the bottom right one of the two red degrees of freedom
 *   is $x_{12}$ and that the two yellow neighbors on its left and right
 *   are $x_{28}$ and $x_{40}$. Then, requiring that the finite element
 *   function be continuous is equivalent to requiring that $x_{12}=
 *   \frac 12 (x_{28}+x_{40})$. A similar situation occurs in the
 *   context of hp adaptive finite element methods.
 *   For example, when using Q1 and Q2 elements (i.e. using
 *   FE_Q(1) and FE_Q(2)) on the two marked cells of the mesh
 *       @image html hp-refinement-simple.png
 *   there are three constraints: first $x_2=\frac 12 x_0 + \frac 12 x_1$,
 *   then $x_4=\frac 14 x_0 + \frac 34 x_1$, and finally the identity
 *   $x_3=x_1$. Similar constraints occur as hanging nodes even if all
 *   cells used the same finite elements. In all of these cases, you
 *   would use the DoFTools::make_hanging_node_constraints function to
 *   compute such constraints.
 * - Other linear constraints, for example when you try to impose a certain
 *   average value for a problem that would otherwise not have a unique
 *   solution. An example of this is given in the step-11 tutorial program.
 *
 * In all of these examples, constraints on degrees of freedom are linear
 * and possibly inhomogeneous. In other words, the always have
 * the form $x_{i_1} = \sum_{j=2}^M a_{i_j} x_{i_j} + b_i$. The deal.II
 * class that deals with storing and using these constraints is
 * ConstraintMatrix. The naming stems from the fact that the class
 * originally only stored the (sparse) matrix $a_{i_j}$. The class name
 * component "matrix" no longer makes much sense today since the class has
 * learned to also deal with inhomogeneities $b_i$.
 *
 *
 * <h3>Eliminating constraints</h3>
 *
 * When building the global system matrix and the right hand sides, one can
 * build them without taking care of the constraints, i.e. by simply looping
 * over cells and adding the local contributions to the global matrix and
 * right hand side objects. In order to do actual calculations, you have to
 * 'condense' the linear system: eliminate constrained degrees of freedom and
 * distribute the appropriate values to the unconstrained dofs. This changes
 * the sparsity pattern of the sparse matrices used in finite element
 * calculations and is thus a quite expensive operation. The general scheme of
 * things is then that you build your system, you eliminate (condense) away
 * constrained nodes using the ConstraintMatrix::condense() functions, then
 * you solve the remaining system, and finally you compute the values of
 * constrained nodes from the values of the unconstrained ones using the
 * ConstraintMatrix::distribute() function. Note that the
 * ConstraintMatrix::condense() function is applied to matrix and right hand
 * side of the linear system, while the ConstraintMatrix::distribute()
 * function is applied to the solution vector. This is the method used in
 * the first few tutorial programs, see for example step-6.
 *
 * This scheme of first building a linear system and then eliminating
 * constrained degrees of freedom is inefficient, and a bottleneck if there
 * are many constraints and matrices are full, i.e. especially for 3d and/or
 * higher order or hp finite elements. Furthermore, it is impossible to
 * implement for %parallel computations where a process may not have access
 * to elements of the matrix. We therefore offer a second way of
 * building linear systems, using the
 * ConstraintMatrix::add_entries_local_to_global() and
 * ConstraintMatrix::distribute_local_to_global() functions discussed
 * below. The resulting linear systems are equivalent to those one gets after
 * calling the ConstraintMatrix::condense() functions.
 *
 * @note Both ways of applying constraints set the value of the matrix
 * diagonals to constrained entries to a <i>positive</i> entry of the same
 * magnitude as the other entries in the matrix. As a consequence, you need to
 * set up your problem such that the weak form describing the main matrix
 * contribution is not <i>negative definite</i>. Otherwise, iterative solvers
 * such as CG will break down or be considerably slower as GMRES.
 *
 * @note While these two ways are <i>equivalent</i>, i.e., the solution of
 * linear systems computed via either approach is the same, the linear
 * systems themselves do not necessarily have the same matrix and right
 * hand side vector entries. Specifically, the matrix diagonal and right hand
 * side entries corresponding to constrained degrees of freedom may be different
 * as a result of the way in which we compute them; they are, however, always
 * chosen in such a way that the solution to the linear system is the same.
 *
 * <h4>Condensing matrices and sparsity patterns</h4>
 *
 * As mentioned above, the first way of using constraints is to build linear
 * systems without regards to constraints and then "condensing" them away.
 * Condensation of a matrix is done in four steps:
 * - first one builds the
 *   sparsity pattern (e.g. using DoFTools::make_sparsity_pattern());
 * - then the sparsity pattern of the condensed matrix is made out of
 *   the original sparsity pattern and the constraints;
 * - third, the global matrix is assembled;
 * - and fourth, the matrix is finally condensed.
 *
 * To do these steps, you have (at least) two possibilities:
 *
 * <ul>
 * <li> Use two different sparsity patterns and two different matrices: you
 * may eliminate the rows and columns associated with a constrained degree
 * of freedom, and create a
 * totally new sparsity pattern and a new system matrix. This has the
 * advantage that the resulting system of equations is smaller and free from
 * artifacts of the condensation process and is therefore faster in the
 * solution process since no unnecessary multiplications occur (see
 * below). However, there are two major drawbacks: keeping two matrices at the
 * same time can be quite unacceptable if you're short of memory. Secondly,
 * the condensation process is expensive, since <em>all</em> entries of the
 * matrix have to be copied, not only those which are subject to constraints.
 *
 * This procedure is therefore not advocated and not discussed in the @ref
 * Tutorial. deal.II used to have functions that could perform these shrinking
 * operations, but for the reasons outlined above they were inefficient, rarely
 * used and consequently removed in version 8.0.
 *
 * <li> Use only one sparsity pattern and one matrix: doing it this way, the
 * condense functions add nonzero entries to the sparsity pattern of the large
 * matrix (with constrained nodes in it) where the condensation process of the
 * matrix will create additional nonzero elements. In the condensation process
 * itself, rows and columns subject to constraints are distributed to the rows
 * and columns of unconstrained nodes. The constrained degrees of freedom
 * remain in place,
 * however, unlike in the first possibility described above. In order not to
 * disturb the solution process, these rows and columns are filled with zeros
 * and an appropriate positive value on the main diagonal (we choose an
 * average of the magnitudes of the other diagonal elements, so as to make
 * sure that the new diagonal entry has the same order of magnitude as the
 * other entries; this preserves the scaling properties of the matrix). The
 * corresponding value in the right hand sides is set to zero. This way, the
 * constrained node will always get the value zero upon solution of the
 * equation system and will not couple to other nodes any more.
 *
 * This method has the advantage that only one matrix and sparsity pattern is
 * needed thus using less memory. Additionally, the condensation process is
 * less expensive, since not all but only constrained values in the matrix
 * have to be copied. On the other hand, the solution process will take a bit
 * longer, since matrix vector multiplications will incur multiplications with
 * zeroes in the lines subject to constraints. Additionally, the vector size
 * is larger than in the first possibility, resulting in more memory
 * consumption for those iterative solution methods using a larger number of
 * auxiliary vectors (e.g. methods using explicit orthogonalization
 * procedures).
 *
 * Nevertheless, this process is overall more efficient due to its lower
 * memory consumption and is the one discussed in the first few programs
 * of the @ref Tutorial , for example in step-6.
 * </ul>
 *
 * The ConstraintMatrix class provides two sets of @p condense functions:
 * those taking two arguments refer to the first possibility above, those
 * taking only one do their job in-place and refer to the second possibility.
 *
 * The condensation functions exist for different argument types. The
 * in-place functions (i.e. those following the second way) exist for
 * arguments of type SparsityPattern, SparseMatrix and
 * BlockSparseMatrix. Note that there are no versions for arguments of type
 * PETScWrappers::SparseMatrix() or any of the other PETSc or Trilinos
 * matrix wrapper classes. This is due to the fact that it is relatively
 * hard to get a representation of the sparsity structure of PETSc matrices,
 * and to modify them; this holds in particular, if the matrix is actually
 * distributed across a cluster of computers. If you want to use
 * PETSc/Trilinos matrices, you can either copy an already condensed deal.II
 * matrix, or assemble the PETSc/Trilinos matrix in the already condensed form,
 * see the discussion below.
 *
 *
 * <h5>Condensing vectors</h5>
 *
 * Condensing vectors works exactly as described above for matrices. Note that
 * condensation is an idempotent operation, i.e. doing it more than once on a
 * vector or matrix yields the same result as doing it only once: once an
 * object has been condensed, further condensation operations don't change it
 * any more.
 *
 * In contrast to the matrix condensation functions, the vector condensation
 * functions exist in variants for PETSc and Trilinos vectors. However,
 * using them is typically expensive, and should be avoided. You should use
 * the same techniques as mentioned above to avoid their use.
 *
 *
 * <h4>Avoiding explicit condensation</h4>
 *
 * Sometimes, one wants to avoid explicit condensation of a linear system
 * after it has been built at all. There are two main reasons for wanting to
 * do so:
 *
 * <ul>
 * <li>
 * Condensation is an expensive operation, in particular if there
 * are many constraints and/or if the matrix has many nonzero entries. Both
 * is typically the case for 3d, or high polynomial degree computations, as
 * well as for hp finite element methods, see for example the @ref hp_paper
 * "hp paper". This is the case discussed in the hp tutorial program, @ref
 * step_27 "step-27", as well as in step-22 and @ref step_31
 * "step-31".
 *
 * <li>
 * There may not be a ConstraintMatrix::condense() function for the matrix
 * you use (this is, for example, the case for the PETSc and Trilinos
 * wrapper classes where we have no access to the underlying representation
 * of the matrix, and therefore cannot efficiently implement the
 * ConstraintMatrix::condense() operation). This is the case discussed
 * in step-17, step-18, step-31, and step-32.
 * </ul>
 *
 * In this case, one possibility is to distribute local entries to the final
 * destinations right at the moment of transferring them into the global
 * matrices and vectors, and similarly build a sparsity pattern in the
 * condensed form at the time it is set up originally.
 *
 * The ConstraintMatrix class offers support for these operations as well. For
 * example, the ConstraintMatrix::add_entries_local_to_global() function adds
 * nonzero entries to a sparsity pattern object. It not only adds a given
 * entry, but also all entries that we will have to write to if the current
 * entry corresponds to a constrained degree of freedom that will later be
 * eliminated. Similarly, one can use the
 * ConstraintMatrix::distribute_local_to_global() functions to directly
 * distribute entries in vectors and matrices when copying local contributions
 * into a global matrix or vector. These calls make a subsequent call to
 * ConstraintMatrix::condense() unnecessary. For examples of their use see the
 * tutorial programs referenced above.
 *
 * Note that, despite their name which describes what the function really
 * does, the ConstraintMatrix::distribute_local_to_global() functions has to
 * be applied to matrices and right hand side vectors, whereas the
 * ConstraintMatrix::distribute() function discussed below is applied to the
 * solution vector after solving the linear system.
 *
 *
 * <h3>Distributing constraints</h3>
 *
 * After solving the condensed system of equations, the solution vector has to
 * be "distributed": the modification to the original linear system that
 * results from calling ConstraintMatrix::condense leads to a linear system
 * that solves correctly for all degrees of freedom that are unconstrained but
 * leaves the values of constrained degrees of freedom undefined. To get the
 * correct values also for these degrees of freedom, you need to "distribute"
 * the unconstrained values also to their constrained colleagues. This is done
 * by the two ConstraintMatrix::distribute() functions, one working with two
 * vectors, one working in-place. The operation of distribution undoes the
 * condensation process in some sense, but it should be noted that it is not
 * the inverse operation. Basically, distribution sets the values of the
 * constrained nodes to the value that is computed from the constraint given
 * the values of the unconstrained nodes plus possible inhomogeneities.
 *
 *
 * <h3>Treatment of inhomogeneous constraints</h3>
 *
 * In case some constraint lines have inhomogeneities (which is typically the
 * case if the constraint comes from implementation of inhomogeneous boundary
 * conditions), the situation is a bit more complicated than if the only
 * constraints were due to hanging nodes alone. This is because the
 * elimination of the non-diagonal values in the matrix generate contributions
 * in the eliminated rows in the vector. This means that inhomogeneities can
 * only be handled with functions that act simultaneously on a matrix and a
 * vector. This means that all inhomogeneities are ignored in case the
 * respective condense function is called without any matrix (or if the matrix
 * has already been condensed before).
 *
 * The use of ConstraintMatrix for implementing Dirichlet boundary conditions
 * is discussed in the step-22 tutorial program. A further example that applies
 * the ConstraintMatrix is step-41. The situation here is little more complicated,
 * because there we have some constraints which are not at the boundary.
 * There are two ways to apply inhomogeneous constraints after creating the
 * ConstraintMatrix:
 *
 * First approach:
 * - Apply the ConstraintMatrix::distribute_local_to_global() function to the
 *   system matrix and the right-hand-side with the parameter
 *   use_inhomogeneities_for_rhs = false (i.e., the default)
 * - Set the solution to zero in the inhomogeneous constrained components
 *   using the ConstraintMatrix::set_zero() function (or start with a solution
 *   vector equal to zero)
 * - solve() the linear system
 * - Apply ConstraintMatrix::distribute() to the solution
 *
 * Second approach:
 * - Use the ConstraintMatrix::distribute_local_to_global() function with the parameter
 *   use_inhomogeneities_for_rhs = true and apply it to
 *   the system matrix and the right-hand-side
 * - Set the concerning components of the solution to the inhomogeneous
 *   constrained values (for example using ConstraintMatrix::distribute())
 * - solve() the linear system
 * - Depending on the solver now you have to apply the ConstraintMatrix::distribute()
 *   function to the solution, because the solver could change the constrained
 *   values in the solution. For a Krylov based solver this should not be strictly
 *   necessary, but it is still possible that there is a difference between the
 *   inhomogeneous value and the solution value in the order of machine precision,
 *   and you may want to call ConstraintMatrix::distribute() anyway if you have
 *   additional constraints such as from hanging nodes.
 *
 * Of course, both approaches lead to the same final answer but in different
 * ways. Using approach (i.e., when using use_inhomogeneities_for_rhs = false
 * in ConstraintMatrix::distribute_local_to_global()), the linear system we
 * build has zero entries in the right hand side in all those places where a
 * degree of freedom is constrained, and some positive value on the matrix
 * diagonal of these lines. Consequently, the solution vector of the linear
 * system will have a zero value for inhomogeneously constrained degrees of
 * freedom and we need to call ConstraintMatrix::distribute() to give these
 * degrees of freedom their correct nonzero values.
 *
 * On the other hand, in the second approach, the matrix diagonal element and
 * corresponding right hand side entry for inhomogeneously constrained degrees
 * of freedom are so that the solution of the linear system already has the
 * correct value (e.g., if the constraint is that $x_{13}=42$ then row $13$ if
 * the matrix is empty with the exception of the diagonal entry, and
 * $b_{13}/A_{13,13}=42$ so that the solution of $Ax=b$ must satisfy
 * $x_{13}=42$ as desired). As a consequence, we do not need to call
 * ConstraintMatrix::distribute() after solving to fix up inhomogeneously
 * constrained components of the solution, though there is also no harm in
 * doing so.
 *
 * There remains the question of which of the approaches to take and why we
 * need to set to zero the values of the solution vector in the first
 * approach. The answer to both questions has to do with how iterative solvers
 * solve the linear system. To this end, consider that we typically stop
 * iterations when the residual has dropped below a certain fraction of the
 * norm of the right hand side, or, alternatively, a certain fraction of the
 * norm of the initial residual. Now consider this:
 *
 * - In the first approach, the right hand side entries for constrained
 *   degrees of freedom are zero, i.e., the norm of the right hand side
 *   really only consists of those parts that we care about. On the other
 *   hand, if we start with a solution vector that is not zero in
 *   constrained entries, then the initial residual is very large because
 *   the value that is currently in the solution vector does not match the
 *   solution of the linear system (which is zero in these components).
 *   Thus, if we stop iterations once we have reduced the initial residual
 *   by a certain factor, we may reach the threshold after a single
 *   iteration because constrained degrees of freedom are resolved by
 *   iterative solvers in just one iteration. If the initial residual
 *   was dominated by these degrees of freedom, then we see a steep
 *   reduction in the first step although we did not really make much
 *   progress on the remainder of the linear system in this just one
 *   iteration. We can avoid this problem by either stopping iterations
 *   once the norm of the residual reaches a certain fraction of the
 *   <i>norm of the right hand side</i>, or we can set the solution
 *   components to zero (thus reducing the initial residual) and iterating
 *   until we hit a certain fraction of the <i>norm of the initial
 *   residual</i>.
 * - In the second approach, we get the same problem if the starting vector
 *   in the iteration is zero, since then then the residual may be
 *   dominated by constrained degrees of freedom having values that do not
 *   match the values we want for them at the solution. We can again
 *   circumvent this problem by setting the corresponding elements of the
 *   solution vector to their correct values, by calling
 *   ConstraintMatrix::distribute() <i>before</i> solving the linear system
 *   (and then, as necessary, a second time after solving).
 *
 * In addition to these considerations, consider the case where we have
 * inhomogeneous constraints of the kind $x_{3}=\tfrac 12 x_1 + \tfrac 12$,
 * e.g., from a hanging node constraint of the form $x_{3}=\tfrac 12 (x_1 +
 * x_2)$ where $x_2$ is itself constrained by boundary values to $x_2=1$.
 * In this case, the ConstraintMatrix can of course not figure out what
 * the final value of $x_3$ should be and, consequently, can not set the
 * solution vector's third component correctly. Thus, the second approach will
 * not work and you should take the first.
 *
 *
 * <h3>Dealing with conflicting constraints</h3>
 *
 * There are situations where degrees of freedom are constrained in more
 * than one way, and sometimes in conflicting ways. Consider, for example
 * the following situation:
 *     @image html conflicting_constraints.png ""
 * Here, degree of freedom $x_0$ marked in blue is a hanging node. If we
 * used trilinear finite elements, i.e. FE_Q(1), then it would carry the
 * constraint $x_0=\frac 12 (x_{1}+x_{2})$. On the other hand, it is at
 * the boundary, and if we have imposed boundary conditions
 * $u|_{\partial\Omega}=g$ then we will have the constraint $x_0=g_0$
 * where $g_0$ is the value of the boundary function $g(\mathbf x)$ at
 * the location of this degree of freedom.
 *
 * So, which one will win? Or maybe: which one <i>should</i> win? There is
 * no good answer to this question:
 * - If the hanging node constraint is the one that is ultimately enforced,
 *   then the resulting solution does not satisfy boundary
 *   conditions any more for general boundary functions $g$.
 * - If it had been done the other way around, the solution would not satisfy
 *   hanging node constraints at this point and consequently would not
 *   satisfy the regularity properties of the element chosen (e.g. would not
 *   be continuous despite using a $Q_1$ element).
 * - The situation becomes completely hopeless if you consider
 *   curved boundaries since then the edge midpoint (i.e. the hanging node)
 *   does in general not lie on the mother edge. Consequently, the solution
 *   will not be $H^1$ conforming anyway, regardless of the priority of
 *   the two competing constraints. If the hanging node constraint wins, then
 *   the solution will be neither conforming, nor have the right boundary
 *   values.
 * In other words, it is not entirely clear what the "correct" solution would
 * be. In most cases, it will not matter much: in either case, the error
 * introduced either by the non-conformity or the incorrect boundary values
 * will be at worst at the same order as the discretization's overall error.
 *
 * That said, what should you do if you know what you want is this:
 * - If you want the hanging node constraints to win, then first build
 *   these through the DoFTools::make_hanging_node_constraints() function.
 *   Then interpolate the boundary values using
 *   VectorTools::interpolate_boundary_values() into the same ConstraintMatrix
 *   object. If the latter function encounters a boundary node that already
 *   is constrained, it will simply ignore the boundary values at this
 *   node and leave the constraint untouched.
 * - If you want the boundary value constraint to win, build the hanging
 *   node constraints as above and use these to assemble the matrix using
 *   the ConstraintMatrix::distribute_local_to_global() function (or,
 *   alternatively, assemble the matrix and then use
 *   ConstraintMatrix::condense() on it). In a second step, use the
 *   VectorTools::interpolate_boundary_values() function that returns
 *   a std::map and use it as input for MatrixTools::apply_boundary_values()
 *   to set boundary nodes to their correct value.
 *
 * Either behavior can also be achieved by building two separate
 * ConstraintMatrix objects and calling ConstraintMatrix::merge function with
 * a particular second argument.
 */
