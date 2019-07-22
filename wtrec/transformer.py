from pandas import read_html, DataFrame
import numpy as np
import re
import imgkit
from bs4 import BeautifulSoup as bs
from joblib import Parallel, delayed
from tqdm import tqdm_notebook as tqdm
import PIL
from tempfile import NamedTemporaryFile


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
                f'ratio_select_{idx}': len([1 for x in i[0] if len(bs(x, 'html.parser').find_all('select'))]) / len(i[0]),
                f'ratio_f_{idx}': len([1 for x in i[0] if len(bs(x, 'html.parser').find_all(['b', 'u', 'font', 'i']))]) / len(i[0]),
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
        new_records.append(_BaselineSample(rec).transform())

    Parallel(n_jobs=-1, require='sharedmem')(delayed(_transform)(i) for i in tqdm(records))

    return DataFrame(new_records)


class _ApproachSample(object):
    def __init__(self,
                 obj,
                 render_field='transformed_html',
                 strategy=None,
                 scale_cell_dimensions=True,
                 cell_size='5px',
                 long_text_threshold=10,
                 draw_borders=True,
                 target_shape=(224, 224),
                 resize_mode='stretch'):
        super().__init__()
        self.obj = obj
        self.render_field = render_field
        self.strategy = strategy
        self.scale_cell_dimensions = scale_cell_dimensions
        self.cell_size = cell_size
        self.long_text_threshold = long_text_threshold
        self.draw_borders = draw_borders
        self.target_shape = target_shape
        self.resize_mode = resize_mode

    def _preprocess_html_grid(self):
        soup = bs(self.obj['raw'], 'html.parser')

        soup = self._clear_styling_attributes(soup)

        for tag in soup.find_all(['th','td']):
            if self.scale_cell_dimensions:
                tag['width'] = self.cell_size
                tag['height'] = self.cell_size

            color = 'white'
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
                elif len(text) > self.long_text_threshold:
                    color = 'brown'
                elif tag.find('b'):
                    color = 'orange'
                else:
                    color = 'yellow'

            tag['style'] = 'background-color: ' + color
            # replace content
                # ALTERNATIVE CODE INCASE WE DECIDE TO KEEP THE STRUCTURE
                # if KEEP_STRUCTURE and tag.string:
                #   tag.string = "&nbsp;" * len(tag.string.strip())
            tag.clear()

        if not self.draw_borders:
            tag = soup.find('table')
            tag['cellspacing'] = 0
            tag['cellpadding'] = 0

        self.obj.update({
            'transformed_html': str(soup.prettify(formatter='minimal'))
        })

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

    def _generate_image_from_html(self, html):
        with NamedTemporaryFile(suffix='.png') as f:
            try:  # tables containing iframes or similar external sources cannot be rendered
                imgkit.from_string(html,
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
        if self.resize_mode == 'resize':
            canvas = PIL.Image.new('RGB', self.target_shape, color=(255, 255, 255))
            canvas.paste(image)
            image.thumbnail(self.target_shape, PIL.Image.ANTIALIAS)
        elif self.resize_mode == 'stretch':
            image = image.resize(self.target_shape, PIL.Image.ANTIALIAS)
        elif self.resize_mode == 'crop':
            canvas = PIL.Image.new('RGB', self.target_shape, color=(255, 255, 255))
            canvas.paste(image)
            image = image.crop((0, 0, self.target_shape[0], self.target_shape[1]))

        return image

    def _render_html(self):
        image = self._generate_image_from_html(self.obj[self.render_field])  # .decode('utf-8', 'replace'))
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
        try:
            if self.strategy == 'grid':
                self._preprocess_html_grid()
            elif self.strategy == 'char_blocks':
                self.obj['transformed_html'] = '<table></table>'  # TODO implement
            elif self.strategy == 'color_shades':
                self._preprocess_html_color_shades()
        except:
            self.obj['transformed_html'] = '<table></table>'

        self._render_html()

        return self.obj


def transform_for_approach(raw_dataframe, strategy='grid'):
    """
    Transform an unprocessed web table dataset to feature space according to our approach

    Args:
        Dataframe with columns raw and label
    Returns:
        Dataframe with columns raw, label, transformed_html and image
        Generates image representations of web table
    """
    records = raw_dataframe.to_dict('records')
    new_records = []

    def _transform(rec):
        new_records.append(_ApproachSample(rec, strategy=strategy).transform())

    Parallel(n_jobs=-1, require='sharedmem')(delayed(_transform)(i) for i in tqdm(records))

    return DataFrame(new_records)
