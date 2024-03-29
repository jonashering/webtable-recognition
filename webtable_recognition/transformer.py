from tempfile import NamedTemporaryFile
from unicodedata import normalize

import numpy as np
import PIL
import imgkit
import regex as re
from pandas import read_html, DataFrame
from bs4 import BeautifulSoup as bs
from joblib import Parallel, delayed
from tqdm import tqdm_notebook as tqdm


class _BaselineSample(object):
    def __init__(self, obj):
        super().__init__()
        self.obj = obj
        self.raw = str(bs(self.obj['raw'], 'html.parser').find_all('table')[0])
        self.as_df = read_html(self.raw)[0].fillna('')

    def _load_row_html(self, idx):
        row = ''
        try:
            row = bs(self.raw, 'html.parser').find_all('tr')[idx]
        except IndexError:
            row = bs(self.raw, 'html.parser').find_all('tr')[-1]
        cells = []
        for cell in row.find_all(['td', 'th']):
            cells.append(str(cell))

        return cells

    def _load_row_clean(self, idx):
        try:
            row = self.as_df.iloc[idx, :]
        except IndexError:
            row = self.as_df.iloc[-1, :]

        return row

    def _load_col_html(self, idx):
        col = bs(self.raw, 'html.parser').find_all('tr')
        cells = []
        for row in col:
            cell = ''
            try:
                cell = row.find_all(['td', 'th'])[idx]
            except IndexError:
                cell = row.find_all(['td', 'th'])[-1]
            cells.append(str(cell))

        return cells

    def _load_col_clean(self, idx):
        try:
            col = self.as_df.iloc[:, idx]
        except IndexError:
            col = self.as_df.iloc[: -1]

        return col

    def _parse(self):
        self.as_df = read_html(self.raw)[0].fillna('')
        self.rows = [
            (self._load_row_html(0), self._load_row_clean(0)),
            (self._load_row_html(1), self._load_row_clean(1)),
            (self._load_row_html(self.as_df.shape[0] - 1), self._load_row_clean(self.as_df.shape[0] - 1))
        ]
        self.cols = [
            (self._load_col_html(0), self._load_col_clean(0)),
            (self._load_col_html(1), self._load_col_clean(1)),
            (self._load_col_html(self.as_df.shape[1] - 1), self._load_col_clean(self.as_df.shape[1] - 1))
        ]

    def _add_global_layout_features(self):
        features = {
            'max_rows': self.as_df.shape[0],
            'max_cols': self.as_df.shape[1],
            'max_cell_length': max([len(str(elem)) for elem in np.array(self.as_df).flatten()]),
        }

        self.obj.update(features)

    def _add_layout_features(self):
        for idx, i in enumerate(self.cols):
            total_rowspan = np.sum(
                [int(bs(x, 'html.parser').find_all(['td', 'th'])[0].attrs.get('rowspan', 0)) for x in i[0]]
            )
            num_rowspan = len([1 for x in i[0] if 'rowspan' in bs(x, 'html.parser').find_all(['td', 'th'])[0].attrs])
            features = {
                f'avg_length_{idx}': np.mean([len(str(elem)) for elem in i[1]]),
                f'length_variance_{idx}': np.var([len(str(elem)) for elem in i[1]]),
                f'ratio_colspan_{idx}': 0,  # this is a row!
                f'ratio_rowspan_{idx}': (total_rowspan - num_rowspan) / len(i[1])
            }

            self.obj.update(features)

        for idx, i in enumerate(self.rows):
            total_colspan = np.sum(
                [int(bs(x, 'html.parser').find_all(['td', 'th'])[0].attrs.get('colspan', 0)) for x in i[0]]
            )
            num_colspan = len([1 for x in i[0] if 'colspan' in bs(x, 'html.parser').find_all(['td', 'th'])[0].attrs])

            features = {
                f'avg_length_{idx}': np.mean([len(str(elem)) for elem in i[1]]),
                f'length_variance_{idx}': np.var([len(str(elem)) for elem in i[1]]),
                f'ratio_colspan_{idx}': (total_colspan - num_colspan) / len(i[1]),
                f'ratio_rowspan_{idx}': 0
            }

            self.obj.update(features)

    def _add_html_features(self):
        for idx, i in enumerate(self.rows + self.cols):
            features = {
                f'dist_tags_{idx}': len([1 for x in i[0] if len(bs(x, 'html.parser').find_all('br'))]) / len(i[0]),
                f'ratio_th_{idx}': len([1 for x in i[0] if len(bs(x, 'html.parser').find_all('th'))]) / len(i[0]),
                f'ratio_anchor_{idx}': len([1 for x in i[0] if len(bs(x, 'html.parser').find_all('a'))]) / len(i[0]),
                f'ratio_img_{idx}': len([1 for x in i[0] if len(bs(x, 'html.parser').find_all('img'))]) / len(i[0]),
                f'ratio_input_{idx}': len([1 for x in i[0] if len(bs(x, 'html.parser').find_all('input'))]) / len(i[0]),
                f'ratio_select_{idx}':
                    len([1 for x in i[0] if len(bs(x, 'html.parser').find_all('select'))]) / len(i[0]),
                f'ratio_f_{idx}':
                    len([1 for x in i[0] if len(bs(x, 'html.parser').find_all(['b', 'u', 'font', 'i']))]) / len(i[0]),
                f'ratio_br_{idx}': len([1 for x in i[0] if len(bs(x, 'html.parser').find_all('br'))]) / len(i[0]),
            }

            self.obj.update(features)

    def _add_lexical_features(self):
        for idx, i in enumerate(self.rows + self.cols):
            features = {
                f'dist_string_{idx}': len(list(set([re.sub(r'\b\d+\b', '', str(x)) for x in i[1]]))) / len(i[1]),
                f'ratio_colon_{idx}': np.mean([int(str(x).endswith(':')) for x in i[1]]),
                f'ratio_contain_number_{idx}': np.mean([int(any(char.isdigit() for char in str(x))) for x in i[1]]),
                f'ratio_is_number_{idx}': np.mean([int(type(x) in ['float', 'int']) for x in i[1]]),
                f'ratio_nonempty_{idx}': np.mean([int(len(str(x)) > 0) for x in i[1]]),
            }

            self.obj.update(features)

    def transform(self):
        """
        Generate feature vector for a single web table according to baseline

        Args:
            None
        Returns:
            Dataframe with raw, label and feture vector for a single web column
        """
        self._parse()
        self._add_global_layout_features()
        self._add_layout_features()
        self._add_html_features()
        self._add_lexical_features()

        return self.obj


def transform_for_baseline(raw_dataframe):
    """
    Transform an unprocessed web table dataset to feature space according to baseline

    Args:
        Dataframe with columns raw and label
    Returns:
        Dataframe with columns raw, label and feature space (107 columns)
    """
    records = raw_dataframe.to_dict('records')
    new_records = []

    def _transform(rec):
        try:
            new_records.append(_BaselineSample(rec).transform())
        except:
            print('Skip ', rec['path'])

    Parallel(n_jobs=-1, require='sharedmem')(delayed(_transform)(i) for i in tqdm(records))

    return DataFrame(new_records)


class _ApproachSample(object):
    def __init__(self,
                 obj,
                 strategy=None,
                 scale_cell_dimensions=True,
                 cell_size='5px',
                 long_text_threshold=10,
                 use_long_text_threshold=False,
                 remove_borders=False,
                 target_shape=(224, 224),
                 resize_mode='stretch'):
        super().__init__()
        self.obj = obj
        self.strategy = strategy
        self.scale_cell_dimensions = scale_cell_dimensions
        self.cell_size = cell_size
        self.long_text_threshold = long_text_threshold
        self.use_long_text_threshold = use_long_text_threshold
        self.remove_borders = remove_borders
        self.target_shape = target_shape
        self.resize_mode = resize_mode

    def _clear_styling_attributes(self, soup):
        # clear all attributes that could impact styling (except col- and rowspan)
        for tag in soup.find_all():
            new_attr = {}
            if 'colspan' in tag.attrs:
                new_attr['colspan'] = tag.attrs['colspan']
            if 'rowspan' in tag.attrs:
                new_attr['rowspan'] = tag.attrs['rowspan']
            tag.attrs = new_attr
        return soup

    def _scale_cell_dimensions(self, tag):
        if self.scale_cell_dimensions:
            tag['width'] = self.cell_size
            tag['height'] = self.cell_size
        return tag

    def _remove_borders(self, soup):
        if self.remove_borders:
            tag = soup.find('table')
            tag['cellspacing'] = 0
            tag['cellpadding'] = 0
        return soup

    def _is_emphasized(self, tag):
        return len(tag.find_all(['b', 'strong', 'i'])) > 0

    def _preprocess_html_color_shades(self):
        soup = bs(self.obj['raw'], 'html.parser')

        soup = self._clear_styling_attributes(soup)

        for tag in soup.find_all(['th', 'td']):
            tag = self._scale_cell_dimensions(tag)
            text = tag.text.strip()

            # set red for data type
            # set r_step so there is an equivalent distance between the groups (255 / 6 ~= 42)
            r_step = 42
            r = 0 * r_step
            if tag.find('a'):
                r = 1 * r_step
            elif tag.find('img'):
                r = 2 * r_step
            elif tag.find('button'):
                r = 3 * r_step
            elif tag.find('form') or tag.find('input'):
                r = 4 * r_step
            elif len(text) > 0:
                # cells text majority are numeric characters
                if sum(c.isdigit() for c in text) > (len(text) / 2):
                    r = 5 * r_step
                else:
                    r = 255

            # set green for content length
            g = min(len(text), 255)

            # set blue for styling
            b = 0
            if self._is_emphasized(tag):
                b = 127
            elif tag.name == 'th':
                b = 255

            tag['style'] = f'background-color: rgb({r},{g},{b})'
            tag.clear()

        soup = self._remove_borders(soup)

        self.obj.update({
            'transformed_html': str(soup.prettify(formatter='minimal'))
        })

    def _preprocess_html_grid(self):
        soup = bs(self.obj['raw'], 'html.parser')

        soup = self._clear_styling_attributes(soup)

        for tag in soup.find_all(['th', 'td']):
            tag = self._scale_cell_dimensions(tag)

            color = 'yellow'
            if tag.name == 'th':
                color = 'grey'
            elif tag.find('a'):
                color = 'blue'
            elif tag.find('img'):
                color = 'green'
            elif tag.find('button'):
                color = 'purple'
            elif tag.find('form') or tag.find('input'):
                color = 'pink'
            else:
                text = tag.text.strip()
                # cells text majority are numeric characters
                if sum(c.isdigit() for c in text) > (len(text) / 2):
                    color = 'red'
                elif self.use_long_text_threshold and len(text) > self.long_text_threshold:
                    color = 'brown'
                elif self._is_emphasized(tag):
                    color = 'orange'

            tag['style'] = f'background-color: {color}'
            # replace content
            # ALTERNATIVE CODE INCASE WE DECIDE TO KEEP THE STRUCTURE
            # if KEEP_STRUCTURE and tag.string:
            #     tag.string = "&nbsp;" * len(tag.string.strip())
            tag.clear()

        soup = self._remove_borders(soup)

        self.obj.update({
            'transformed_html': str(soup.prettify(formatter='minimal'))
        })

    def _preprocess_html_char_blocks(self):
        soup = bs(self.obj['raw'], 'html.parser')

        for cell in soup.find_all():  # table head cells
            if 'style' in cell:
                cell['style'] += ';background-color:none !important'
            else:
                cell['style'] = ';background-color:none !important'

        # replace character classes with block symbol : digits, alphabetical, punctuation, whitespace
        for elem in soup.find_all(text=True):
            content = normalize('NFKD', elem)
            for char in content:
                color = 'white'
                if re.match(r'[\p{N}]', char) is not None:  # digits
                    color = 'red'
                elif re.match(r'[\p{L}]', char) is not None:  # alpha
                    color = 'blue'
                elif re.match(r'[!"\#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]', char) is not None:  # punctuation
                    color = 'green'
                new_char = soup.new_tag('span', style=f'color: {color} !important')
                new_char.string = '█' if re.match(r'[ \t\r\n\v\f]', char) is None else char  # whitespace
                elem.parent.append(new_char)
            elem.replace_with('')

        # images
        for img in soup.find_all('img'):
            img['style'] = img.get('style', '') + ';background-color:yellow !important'

        # emphasized text
        for emp in soup.find_all(['a', 'strong', 'b', 'i', 'u', 'title']):
            emp['style'] = emp.get('style', '') + ';opacity:0.4 !important'

        # table head cells
        for th in soup.find_all('th'):
            th['style'] = th.get('style', '') + ';background-color:grey !important'

        # input elements
        for inp in soup.find_all(['button', 'select', 'input']):
            inp['style'] = inp.get('style', '') + ';background-color:pink !important; border: 0; padding: 5px;'

        # draw table border
        for tab in soup.find_all('table'):
            tab['style'] = 'border-collapse: collapse ! important'
            tab['cellpadding'] = '5'

        # draw table border pt. 2
        for cell in soup.find_all(['th', 'td']):
            cell['style'] = cell.get('style', '') + ';border: 2px solid black !important'

        self.obj.update({
            'transformed_html': str(soup.prettify(formatter='minimal'))
        })

    def _generate_image_from_html(self, html):
        with NamedTemporaryFile(suffix='.png') as f:
            try:  # tables containing iframes or similar external sources cannot be rendered
                imgkit.from_string(f'<meta charset="utf-8">{html}',
                                   f.name,
                                   options={'quiet': '',
                                            'disable-plugins': '',
                                            'no-images': '',
                                            'disable-javascript': '',
                                            'height': 1024,
                                            'width': 1024,
                                            'load-error-handling': 'ignore'})
                image = PIL.Image.open(f.name)
            except:
                image = PIL.Image.new('RGB', self.target_shape, (255, 255, 255))
        return image.convert('RGB')

    def _crop_surrounding_whitespace(self, image):
        bg = PIL.Image.new(image.mode, image.size, (255, 255, 255))
        diff = PIL.ImageChops.difference(image, bg)
        bbox = diff.getbbox()
        if not bbox:
            return image
        return image.crop(bbox)

    def _resize(self, image):
        if self.resize_mode == 'none':
            return image
        if self.resize_mode == 'resize':
            canvas = PIL.Image.new('RGB', self.target_shape, color=(255, 255, 255))
            image.thumbnail(self.target_shape, PIL.Image.ANTIALIAS)
            canvas.paste(image)
        elif self.resize_mode == 'resize_fullwidth':
            canvas = PIL.Image.new('RGB', self.target_shape, color=(255, 255, 255))
            image.thumbnail((self.target_shape[0], 1024), PIL.Image.ANTIALIAS)
            canvas.paste(image)
            canvas = canvas.crop((0, 0, self.target_shape[0], self.target_shape[1]))
        elif self.resize_mode == 'stretch':
            canvas = image.resize(self.target_shape, PIL.Image.ANTIALIAS)
        elif self.resize_mode == 'crop':
            canvas = PIL.Image.new('RGB', self.target_shape, color=(255, 255, 255))
            canvas.paste(image)
            canvas = canvas.crop((0, 0, self.target_shape[0], self.target_shape[1]))

        return canvas

    def _render_html(self):
        image = self._generate_image_from_html(self.obj['transformed_html'])  # .decode('utf-8', 'replace'))
        image = self._crop_surrounding_whitespace(image)
        image = self._resize(image)

        self.obj.update({
            'image': image
        })

    def transform(self):
        """
        Generate image re for a single web table according to our approach

        Args:
            None
        Returns:
            Dataframe with raw, label and feture vector for a single web column
        """
        if self.strategy == 'raw':
            self.obj['transformed_html'] = self.obj['raw']
        elif self.strategy == 'grid':
            self._preprocess_html_grid()
        elif self.strategy == 'char_blocks':
            self._preprocess_html_char_blocks()
        elif self.strategy == 'color_shades':
            self._preprocess_html_color_shades()

        self._render_html()

        return self.obj


def transform_for_approach(raw_dataframe, strategy='raw', resize_mode='stretch'):
    """
    Transform an unprocessed web table dataset to feature space according to our approach

    Args:
        Dataframe with columns raw and label
        strategy: raw, grid, char_blocks, color_shades
        resize_mode: stretch, resize, resize_fullwidth, crop
    Returns:
        Dataframe with columns raw, label, transformed_html and image
        Generates image representations of web table
    """
    records = raw_dataframe.to_dict('records')
    new_records = []

    def _transform(rec):
        try:
            new_records.append(_ApproachSample(rec, strategy=strategy, resize_mode=resize_mode).transform())
        except:
            print('Skip ', rec['path'])

    Parallel(n_jobs=-1, require='sharedmem')(delayed(_transform)(i) for i in tqdm(records))

    return DataFrame(new_records)
