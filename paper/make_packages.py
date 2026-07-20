r"""Build the two submission archives.

  submission_ieee.zip   journal version: self-contained .tex (bibliography inlined, so the
                        publisher's BibTeX run is not needed), ieeeaccess.cls, branding assets
                        written under BOTH letter cases (the class asks for Logo.png /
                        notaglineLogo.png while the official template ships logo.png /
                        notaglinelogo.png - this breaks on case-sensitive servers), and figures.

  submission_arxiv.tar.gz  preprint version: self-contained .tex + figures. No .bbl and no .bib
                        needed because the bibliography is inlined.

Run after `python make_selfcontained.py`.
"""
import io
import os
import tarfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = ["fig1_concept.png", "fig2_convergence.png", "fig3_regime_map.png",
        "fig4_scaling.png", "fig5_erosion.png", "fig6_vs_aer.png"]


def read(p):
    with open(os.path.join(HERE, p), "rb") as f:
        return f.read()


# ---------------------------------------------------------------- IEEE zip
ieee = os.path.join(HERE, "submission_ieee.zip")
with zipfile.ZipFile(ieee, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("main_ieee.tex", read("main_ieee.tex"))
    z.writestr("ieeeaccess.cls", read("ieeeaccess.cls"))
    z.writestr("bullet.png", read("bullet.png"))
    # both letter cases: the class requests capitalised names, the template ships lowercase
    logo = read("logo.png")
    z.writestr("logo.png", logo)
    z.writestr("Logo.png", logo)
    notag = read("notaglinelogo.png")
    z.writestr("notaglinelogo.png", notag)
    z.writestr("notaglineLogo.png", notag)
    z.writestr("jtehmLogo.png", logo)
    for f in FIGS:
        z.writestr("figures/" + f, read(os.path.join("figures", f)))
print("wrote submission_ieee.zip")
with zipfile.ZipFile(ieee) as z:
    for n in sorted(z.namelist()):
        print("   ", n)

# ------------------------------------------------------------- arXiv tar.gz
arx = os.path.join(HERE, "submission_arxiv.tar.gz")
with tarfile.open(arx, "w:gz") as t:
    def add(arcname, data):
        info = tarfile.TarInfo(arcname)
        info.size = len(data)
        info.mtime = 0
        t.addfile(info, io.BytesIO(data))
    add("arxiv_selfcontained.tex", read("arxiv_selfcontained.tex"))
    for f in FIGS:
        add("figures/" + f, read(os.path.join("figures", f)))
print("\nwrote submission_arxiv.tar.gz")
with tarfile.open(arx) as t:
    for n in sorted(t.getnames()):
        print("   ", n)
