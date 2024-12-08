from PlotFour import plot_four
from DataSetLoader import makeSolo, makeCombo, makeExt

solo_loader = makeSolo(workers=2)
for (i, (x, y)) in enumerate(solo_loader, 0):
    plot_four(x.numpy()[0], name="solo")
    print(y[0])
solo_loader = None

# combo_loader = makeCombo()
# for (i, (x, y)) in enumerate(combo_loader, 0):
#     plot_four(x.numpy()[0], name="combo")
# combo_loader = None
#
# ext_loader = makeExt()
# for (i, (x, y)) in enumerate(ext_loader, 0):
#     plot_four(x.numpy()[0], name="ext")
# ext_loader = None
