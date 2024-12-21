from .arxiv import ogb_dataset

__partition__ = "cpu-galvani"


def run_path(p, outfile):
    return ogb_dataset("mag", p, outfile)
