import matplotlib.pyplot as plt
from ipywidgets import Output
import IPython
from plt_overfit import overfit_example, output
plt.style.use('./deeplearning.mplstyle')


# find out why regressional overfitting analuyis is not working here but works in the notebook
plt.close("all")
IPython.display.display(output)
ofit = overfit_example(False)
plt.show()