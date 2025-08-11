Webpage [https://optimai-lab.github.io/LLM-OPT/lectures/intro.html](https://optimai-lab.github.io/LLM-OPT/index.html)

Run locally: first install (only once)
```
pip install jupyter-book
```
then everytime run
```
jupyter-book build . --all 
open _build/html/index.html
```
to check the website on your local machine.


Google doc link:

[https://docs.google.com/document/d/1T_nDZsOlqntZ09aozTEBGTQuEmG40ZCxdZDnK5GNkRA/edit?tab=t.0]


Use different boxes for different purposes:

first, install
 pip install sphinx-proof
then you can try

```{prf:definition} Convex function
:label: def:convex
A function $f$ is convex if $f(\theta x+(1-\theta)y) \le \theta f(x) + (1-\theta)f(y)$.
```

See {prf:ref}`def:convex`.

in a markdown block