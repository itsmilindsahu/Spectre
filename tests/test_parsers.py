import textwrap

import pytest

from spectre.ingestion.parsers import load_spectrum, Spectrum


def test_load_csv(tmp_path):
    csv_content = "wavenumber,absorbance\n4000,0.01\n3000,0.5\n1000,0.9\n"
    f = tmp_path / "spec.csv"
    f.write_text(csv_content)

    spectrum = load_spectrum(f)

    assert isinstance(spectrum, Spectrum)
    assert len(spectrum) == 3
    assert spectrum.wavenumbers[0] == 4000  # descending order preserved


def test_load_csv_no_header(tmp_path):
    csv_content = "4000,0.01\n3000,0.5\n1000,0.9\n"
    f = tmp_path / "spec_noheader.csv"
    f.write_text(csv_content)

    spectrum = load_spectrum(f)
    assert len(spectrum) == 3


def test_unsupported_extension_raises(tmp_path):
    f = tmp_path / "spec.xyz"
    f.write_text("junk")
    with pytest.raises(ValueError):
        load_spectrum(f)


def test_load_minimal_jcamp(tmp_path):
    jdx_content = textwrap.dedent("""\
        ##TITLE=Test Compound
        ##JCAMP-DX=4.24
        ##XUNITS=1/CM
        ##YUNITS=ABSORBANCE
        ##FIRSTX=4000
        ##LASTX=3998
        ##DELTAX=-1
        ##XFACTOR=1
        ##YFACTOR=1
        ##NPOINTS=3
        ##XYDATA=(X++(Y..Y))
        4000 0.01 0.02 0.03
        ##END=
        """)
    f = tmp_path / "spec.jdx"
    f.write_text(jdx_content)

    spectrum = load_spectrum(f)
    assert len(spectrum) == 3
    assert spectrum.metadata["format"] == "jcamp-dx"
