# utils.py

def line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Returns the intersection point of two lines defined by (x1, y1) to (x2, y2) and (x3, y3) to (x4, y4).
    Returns (x, y) tuple of the intersection point or None if they don't intersect.
    """
    # Calculate the determinant
    denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denominator == 0:
        return None  # Lines are parallel

    # Calculate the intersection point using parametric equations
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator

    if 0 <= ua <= 1 and 0 <= ub <= 1:
        # Intersection point lies within both line segments
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (x, y)

    return None  # No intersection within the line segments
