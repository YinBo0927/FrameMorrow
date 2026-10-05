"""Small dependency-free check for the buildless project page."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit
import re

root = Path(__file__).parent
class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.references, self.cases = [], [], []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs: self.ids.append(attrs['id'])
        for key in ('href', 'src', 'poster'):
            if key in attrs: self.references.append(attrs[key])
        if 'data-case' in attrs: self.cases.append(attrs['data-case'])

page = Page()
page.feed((root / 'index.html').read_text())
assert len(page.ids) == len(set(page.ids)), 'Duplicate HTML IDs'
for ref in page.references:
    if ref.startswith('#'):
        assert ref == '#' or ref[1:] in page.ids, f'Missing anchor: {ref}'
    elif not re.match(r'\w+:', ref):
        assert (root / urlsplit(ref).path).is_file(), f'Missing asset: {ref}'
for case in page.cases:
    for variant in ('native', 'framemorrow'):
        for extension in ('mp4', 'jpg'):
            path = root / 'assets' / f'{case}-{variant}.{extension}'
            assert path.is_file() and path.stat().st_size > 0, f'Missing media: {path}'
assert (root / '.nojekyll').is_file()
assert len(page.cases) == 3
print('PASS: references, anchors, unique IDs, 3 paired video cases, GitHub Pages entry point')
