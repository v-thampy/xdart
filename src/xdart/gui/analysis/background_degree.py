"""A fitted background's degree, in the background's own row (NEXT-128; the maintainer, 2026-11-03).

His words, of the Peak Fitter's Background: "there should be more options for background - Polynomial (this can replace
Linear, constant). when polynomial is chosen, another box to choose the polynomial degree should show in the same row.
same with the chebyshev and Template + Linear (replace with template with polynomial) and template + Chebyshev - add
degree here as well"; for Phase Analysis, his "Same in both (Recommended)".

The box is the shared whole-number row (``FormRow``, kind "int": hard rule 14), parked in the background choice's row
before its "?", as the Peaks row parks Auto; the choice's "?" serves it and F1 on it opens the same help. It shows only
for a choice that takes a degree, bounded by that family's range from the headless owner (``display_fit_operations``'s
``PEAK_BACKGROUND_DEGREES`` and ``PHASE_BACKGROUND_DEGREES``), which its tooltip names. Each family keeps its own
degree while the window is open (read from the box when the family changes, so a typed value counts before it
commits); a family not yet shown starts at its default: Polynomial 1, the earlier Linear, so a new window fits what it
did; Chebyshev 3, the earlier fixed degree; a spline's four knots, the fewest it takes. "Template + polynomial" is the
polynomial family: its correction has the polynomial's degree.
"""
from xrd_tools.session.readiness import integer_field_refusal

from xdart.gui.widgets.controls_panel import FormRow

#: each family's degree (a spline's knots) until it is changed
DEFAULT_DEGREES = {"polynomial": 1, "chebyshev": 3, "spline": 4}
#: the box's tooltip, by family, with its bounds
_PURPOSE = {
    "polynomial": "The polynomial's degree, {low} to {high}: 0 is a constant, 1 a straight line (the earlier Linear).",
    "chebyshev": ("The Chebyshev polynomial's degree, {low} to {high}, over the fitted range mapped onto −1 to 1; "
                  "3 was the earlier fixed degree."),
    "spline": "The spline's knots, {low} to {high}, spread evenly over the trace: more knots follow a more curved "
              "background.",
}


def _label(family):
    return "Knots" if family == "spline" else "Degree"


class BackgroundDegree:
    """The degree box of ``choice_row`` (a background combo ``FormRow``) for the families ``degrees`` bounds, as
    ``{family: (low, high)}``; ``path`` is the box's row path, ``topic`` its F1 help topic."""

    def __init__(self, choice_row, path, degrees, *, topic):
        self.choice_row = choice_row
        self._bounds = dict(degrees)
        #: each family's degree while the window is open
        self._degrees = {family: DEFAULT_DEGREES[family] for family in self._bounds}
        #: the family whose degree the box holds (it keeps it while a choice without a degree hides the box)
        self._family = None
        first = next(iter(self._bounds))
        self.row = FormRow(label="Degree", path=path, value=str(self._degrees[first]), kind="int",
                           minimum=self._bounds[first][0], maximum=self._bounds[first][1])
        self.row.label.setMinimumWidth(0)
        self.row.editor.setMaximumWidth(44)  # two digits, as the Pts cluster sits on Int's Axis row
        self.row.set_purpose_tooltip(_PURPOSE[first].format(low=self._bounds[first][0], high=self._bounds[first][1]))
        self.row.set_help_topic(topic)  # F1; the choice row's "?" is the box's
        choice_row.add_trailing_widget(self.row)  # before the row's "?", which stays last
        choice_row.editor.currentIndexChanged.connect(lambda _index: self.sync())
        self.sync()

    def family(self, choice=None):
        """The degree family of ``choice`` (the current one by default): "polynomial" for "template+polynomial", None
        for a choice without a degree."""
        choice = self.choice_row.current_value() if choice is None else choice
        family = str(choice).rpartition("+")[2]
        return family if family in self._bounds else None

    def degree(self):
        """The current choice's degree; None for a choice without one, or for text the box refuses."""
        if self.family() is None or self.row.refusal():
            return None
        return self.row.editor.value()

    def refusal(self):
        """Why the box's text is not the current choice's degree, as the box names it ("" when it is, and for a choice
        without a degree, whose hidden box never refuses a fit)."""
        if self.family() is None or not self.row.refusal():
            return ""
        return f"{self.row.label.text()}: {self.row.refusal()}"

    def degree_refusal(self, choice, degree):
        """Why ``degree`` cannot be ``choice``'s, as its box would name it ("" when it can, or the choice takes none):
        a settings file's, checked before anything applies."""
        family = self.family(choice)
        if family is None:
            return ""
        low, high = self._bounds[family]
        reason = ("is not a whole number" if type(degree) is not int else
                  integer_field_refusal(str(degree), minimum=low, maximum=high))
        return f"{_label(family)}: {degree!r} {reason}" if reason else ""

    def choose(self, choice, degree=None):
        """Show ``choice`` (a combo value) and, for one that takes a degree, ``degree`` (None keeps its family's): a
        settings file's, in that order, whichever order the file names them."""
        combo = self.choice_row.editor
        index = combo.findData(choice)
        if index < 0:
            raise ValueError(f"{self.choice_row.label.text()}: {choice!r} is not one of its choices")
        combo.setCurrentIndex(index)
        self.sync()  # the same index emits nothing
        family = self.family()
        if family is not None and degree is not None:
            self._degrees[family] = int(degree)
            self.row.set_number(int(degree))

    def sync(self):
        """The box for the current choice: its family's degree, bounds, label and tooltip, shown; or hidden."""
        family = self.family()
        if family is not None and family != self._family:
            if self._family is not None and not self.row.refusal():
                self._degrees[self._family] = self.row.editor.value()
            low, high = self._bounds[family]
            self.row.label.setText(_label(family))
            self.row.set_integer_bounds(low, high)
            self.row.set_number(self._degrees[family])
            self.row.set_purpose_tooltip(_PURPOSE[family].format(low=low, high=high))
            self._family = family
        self.row.setVisible(family is not None)
