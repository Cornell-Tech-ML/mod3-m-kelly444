from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Approximates the rate of change (derivative) of a function at a specific point.

    This function estimates how much `f` is changing with respect to one of its inputs,
    by slightly adjusting that input and observing how `f` changes.

    Args:
    ----
       f : A function
           A function that takes several numbers and returns one number.
       *vals : Numbers
           The values to plug into the function `f` to see how it changes.
       arg : int, optional (default=0)
           Which input to focus on when measuring the rate of change.
       epsilon : float, optional (default=1e-6)
           A tiny amount to adjust the selected input to see how it affects the function.

    Returns:
    -------
        float
            The estimated rate of change of `f` at the given point.

    """
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    return (f(*vals1) - f(*vals2)) / (2.0 * epsilon)
    # END ASSIGN 1.1


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Add the gradient (rate of change) of this variable to its total.

        This is used to gather all the gradients during backpropagation.

        Args:
        ----
            x : Any
                The value of the gradient to be added.

        """
        ...

    @property
    def unique_id(self) -> int:
        """Get a unique identifier for this variable.

        Each variable in the graph has a unique ID to keep track of it.

        Returns
        -------
        int
            The unique ID of this variable.

        """
        ...

    def is_leaf(self) -> bool:
        """Check if this variable is a starting point in the computation.

        A "leaf" is a variable that doesn't depend on any other variables.

        Returns
        -------
        bool
            True if this is a leaf, False if it depends on other variables.

        """
        ...

    def is_constant(self) -> bool:
        """Check if this variable's value is fixed and doesn't change.

        A constant variable doesn't change during the calculations.

        Returns
        -------
        bool
            True if the variable is constant, False otherwise.

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Get the variables that this variable depends on.

        These are the variables that contribute to this variable's value.

        Returns
        -------
        Iterable[Variable]
            A list of variables that are used to calculate this variable.

        """
        ...

    def chain_rule(self, d: Any) -> Iterable[Tuple["Variable", Any]]:
        """Calculate how the change in this variable affects its parents.

        This uses the chain rule from calculus to figure out how the rate of change
        of this variable affects its parent variables.

        Args:
        ----
            d : Any
                The rate of change (derivative) of this variable.

        Returns:
        -------
        Iterable[Tuple[Variable, Any]]
            A list of pairs (parent_variable, derivative) for each parent variable.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Create an order of variables so that we process them in the right sequence.

    This function makes sure we process each variable in the order needed,
    starting from the final result and going backward to the starting variables.

    Args:
    ----
        variable : Variable
            The final variable in the computation (usually the result).

    Returns:
    -------
        Iterable[Variable]
            A list of variables in the correct order to calculate gradients, starting from the end.

    """
    order: List[Variable] = []
    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Work backward through the graph to calculate gradients.

    This function starts at the final variable (the result) and uses the chain rule
    to calculate the gradients of each variable that led to the result.

    Args:
    ----
        variable : Variable
            The final variable in the computation (the result).
        deriv : Any
            The gradient (rate of change) of the final variable to be passed back.

    """
    queue = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv
    for var in queue:
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0.0)
                derivatives[v.unique_id] = derivatives[v.unique_id] + d


@dataclass
class Context:
    """Stores information needed during the forward pass of a function to later compute gradients."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store values that will be used later to compute gradients.

        These values are saved during the forward pass, so they can be accessed
        later during backpropagation.

        Args:
        ----
            values : Any
                The values to be saved for future use.

        """
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Retrieve the saved values from the forward pass.

        These are the values we saved earlier, and will be used in backpropagation.

        Returns
        -------
        Tuple[Any, ...]
            The values saved during the forward pass.

        """
        return self.saved_values
