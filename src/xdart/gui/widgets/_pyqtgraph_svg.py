"""pyqtgraph 0.14.0 SVG coordinate backport; remove when adopting a fixed release.

Only correctCoordinates from upstream commit
1b84c2b2d5b6f22b41c91733bc4ace808d14998c (Qt 6.11 standalone Z support).
Copyright (c) 2012 University of North Carolina at Chapel Hill, Luke Campagnola.
Distributed under the MIT license; see licenses/LICENSE-pyqtgraph.
"""
import re
import xml.dom.minidom as xml

import numpy as np
from pyqtgraph import functions as fn
from pyqtgraph.Qt import QtGui


def correctCoordinates(node, defs, item, options):
    # correct the defs in the linearGradient
    for d in defs:
        if d.tagName == "linearGradient":
            # reset "gradientUnits" attribute to SVG default value
            d.removeAttribute("gradientUnits")

            # replace with percentages
            for coord in ("x1", "x2", "y1", "y2"):
                if coord.startswith("x"):
                    denominator = item.boundingRect().width()
                else:
                    denominator = item.boundingRect().height()
                percentage = round(float(d.getAttribute(coord)) * 100 / denominator)
                d.setAttribute(coord, f"{percentage}%")

            # replace stops with percentages
            for child in filter(
                lambda e: isinstance(e, xml.Element) and e.tagName == "stop",
                d.childNodes
            ):
                offset = child.getAttribute("offset")
                try:
                    child.setAttribute("offset", f"{round(float(offset) * 100)}%")
                except ValueError:
                    # offset attribute could not be converted to float
                    # must be one of the other SVG accepted formats
                    continue

    ## Remove transformation matrices from <g> tags by applying matrix to coordinates inside.
    ## Each item is represented by a single top-level group with one or more groups inside.
    ## Each inner group contains one or more drawing primitives, possibly of different types.
    groups = node.getElementsByTagName('g')

    ## Since we leave text unchanged, groups which combine text and non-text primitives must be split apart.
    ## (if at some point we start correcting text transforms as well, then it should be safe to remove this)
    groups2 = []
    for grp in groups:
        subGroups = [grp.cloneNode(deep=False)]
        textGroup = None
        for ch in grp.childNodes[:]:
            if isinstance(ch, xml.Element):
                if textGroup is None:
                    textGroup = ch.tagName == 'text'
                if ch.tagName == 'text':
                    if textGroup is False:
                        subGroups.append(grp.cloneNode(deep=False))
                        textGroup = True
                else:
                    if textGroup is True:
                        subGroups.append(grp.cloneNode(deep=False))
                        textGroup = False
            subGroups[-1].appendChild(ch)
        groups2.extend(subGroups)
        for sg in subGroups:
            node.insertBefore(sg, grp)
        node.removeChild(grp)
    groups = groups2

    for grp in groups:
        matrix = grp.getAttribute('transform')
        match = re.match(r'matrix\((.*)\)', matrix)
        if match is None:
            vals = [1,0,0,1,0,0]
        else:
            vals = [float(a) for a in match.groups()[0].split(',')]
        tr = np.array([[vals[0], vals[2], vals[4]], [vals[1], vals[3], vals[5]]])

        removeTransform = False
        for ch in grp.childNodes:
            if not isinstance(ch, xml.Element):
                continue
            if ch.tagName == 'polyline':
                removeTransform = True
                coords = np.array([[float(a) for a in c.split(',')] for c in ch.getAttribute('points').strip().split(' ')])
                coords = fn.transformCoordinates(tr, coords, transpose=True)
                ch.setAttribute('points', ' '.join([','.join([str(a) for a in c]) for c in coords]))
            elif ch.tagName == 'path':
                removeTransform = True
                oldCoords = ch.getAttribute('d').strip()
                if oldCoords == '':
                    continue
                newCoords = []
                for c in oldCoords.split(' '):
                    tokens = c.split(',')
                    if len(tokens) == 1:
                        # this might be a Z
                        newCoords.append(tokens[0])
                    else:
                        x, y = tokens
                        if x[0].isalpha():
                            t = x[0]
                            x = x[1:]
                        else:
                            t = ''
                        nc = fn.transformCoordinates(tr, np.array([[float(x),float(y)]]), transpose=True)
                        newCoords.append(t+str(nc[0,0])+','+str(nc[0,1]))
                newCoords = ' '.join(newCoords)
                # If coords start with L instead of M, then the entire path will not be rendered.
                # (This can happen if the first point had nan values in it--Qt will skip it on export)
                if newCoords[0] != 'M':
                    newCoords = f'M{newCoords[1:]}'
                ch.setAttribute('d', newCoords)
            elif ch.tagName == 'text':
                removeTransform = False
                ## leave text alone for now. Might need this later to correctly render text with outline.
                # c = np.array([
                #     [float(ch.getAttribute('x')), float(ch.getAttribute('y'))],
                #     [float(ch.getAttribute('font-size')), 0],
                #     [0,0]])
                # c = fn.transformCoordinates(tr, c, transpose=True)
                # ch.setAttribute('x', str(c[0,0]))
                # ch.setAttribute('y', str(c[0,1]))
                # fs = c[1]-c[2]
                # fs = (fs**2).sum()**0.5
                # ch.setAttribute('font-size', str(fs))

                ## Correct some font information
                families = ch.getAttribute('font-family').split(',')
                if len(families) == 1:
                    font = QtGui.QFont(families[0].strip('" '))
                    if font.styleHint() == font.StyleHint.SansSerif:
                        families.append('sans-serif')
                    elif font.styleHint() == font.StyleHint.Serif:
                        families.append('serif')
                    elif font.styleHint() == font.StyleHint.Courier:
                        families.append('monospace')
                    ch.setAttribute('font-family', ', '.join([f if ' ' not in f else '"%s"'%f for f in families]))

            ## correct line widths if needed
            if removeTransform and ch.getAttribute('vector-effect') != 'non-scaling-stroke' and grp.getAttribute('stroke-width') != '':
                w = float(grp.getAttribute('stroke-width'))
                s = fn.transformCoordinates(tr, np.array([[w,0], [0,0]]), transpose=True)
                w = ((s[0]-s[1])**2).sum()**0.5
                ch.setAttribute('stroke-width', str(w))

            # Remove non-scaling-stroke if requested
            if options.get('scaling stroke') is True and ch.getAttribute('vector-effect') == 'non-scaling-stroke':
                ch.removeAttribute('vector-effect')

        if removeTransform:
            grp.removeAttribute('transform')

