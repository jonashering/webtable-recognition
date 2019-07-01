from pandas import read_html, concat, DataFrame
import numpy as np
import re
import os
from bs4 import BeautifulSoup as bs


class _BaselineSample(object):
    def __init__(self, obj):
        super().__init__()
        self.obj = obj
        self.raw = str(bs(self.obj['raw'], 'html.parser').find_all('table')[0])

    def _load_row_html(self, idx):
        row = ''
        try:
            row = bs(self.raw, 'html.parser').find_all('tr')[idx]
        except:
            row = bs(self.raw, 'html.parser').find_all('tr')[-1]
        cells = []
        for cell in row.find_all(['td', 'th']):
            cells.append(str(cell))

        return cells

    def _load_row_clean(self, idx):
        return self.as_df.iloc[idx, :]

    def _load_col_html(self, idx):
        col = bs(self.raw, 'html.parser').find_all('tr')
        cells = []
        for row in col:
            cell = ''
            try:
                cell = row.find_all(['td', 'th'])[idx]
            except:
                cell = row.find_all(['td', 'th'])[-1]
            cells.append(str(cell))

        return cells

    def _load_col_clean(self, idx):
        return self.as_df.iloc[:, idx]

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
        features = DataFrame({
            'max_rows': [self.as_df.shape[0]],
            'max_cols': [self.as_df.shape[1]],
            'max_cell_length': [max([len(str(elem)) for elem in np.array(self.as_df).flatten()])],
        }).T

        self.obj = concat([self.obj, features])

    def _add_layout_features(self):
        for i in self.cols:
            total_rowspan = np.sum(
                [int(bs(x, 'html.parser').find_all(['td', 'th'])[0].attrs.get('rowspan', 0)) for x in i[0]]
            )
            num_rowspan = len([1 for x in i[0] if 'rowspan' in bs(x, 'html.parser').find_all(['td', 'th'])[0].attrs])
            features = DataFrame({
                'avg_length': [np.mean([len(str(elem)) for elem in i[1]])],
                'length_variance': [np.var([len(str(elem)) for elem in i[1]])],
                'ratio_colspan': [0],
                'ratio_rowspan': [(total_rowspan - num_rowspan) / len(i[1])]
            }).T

            self.obj = concat([self.obj, features])

        for i in self.rows:
            total_colspan = np.sum(
                [int(bs(x, 'html.parser').find_all(['td', 'th'])[0].attrs.get('colspan', 0)) for x in i[0]]
            )
            num_colspan = len([1 for x in i[0] if 'colspan' in bs(x, 'html.parser').find_all(['td', 'th'])[0].attrs])
            features = DataFrame({
                'avg_length': [np.mean([len(str(elem)) for elem in i[1]])],
                'length_variance': [np.var([len(str(elem)) for elem in i[1]])],
                'ratio_colspan': [(total_colspan - num_colspan) / len(i[1])],
                'ratio_rowspan': [0]
            }).T

            self.obj = concat([self.obj, features])

    def _add_html_features(self):
        for i in self.rows + self.cols:
            features = DataFrame({
                'dist_tags': [len([1 for x in i[0] if len(bs(x, 'html.parser').find_all('br'))]) / len(i[0])],  # FIXME
                'ratio_th': [len([1 for x in i[0] if len(bs(x, 'html.parser').find_all('th'))]) / len(i[0])],
                'ratio_anchor': [len([1 for x in i[0] if len(bs(x, 'html.parser').find_all('a'))]) / len(i[0])],
                'ratio_img': [len([1 for x in i[0] if len(bs(x, 'html.parser').find_all('img'))]) / len(i[0])],
                'ratio_input': [len([1 for x in i[0] if len(bs(x, 'html.parser').find_all('input'))]) / len(i[0])],
                'ratio_select': [len([1 for x in i[0] if len(bs(x, 'html.parser').find_all('select'))]) / len(i[0])],
                'ratio_f': [len([1 for x in i[0] if len(bs(x, 'html.parser').find_all(['b', 'u', 'font', 'i']))]) / len(i[0])],
                'ratio_br': [len([1 for x in i[0] if len(bs(x, 'html.parser').find_all('br'))]) / len(i[0])],
            }).T

            self.obj = concat([self.obj, features])

    def _add_lexical_features(self):
        for i in self.rows + self.cols:
            features = DataFrame({
                'dist_string': [len(list(set([re.sub(r'\b\d+\b', '', str(x)) for x in i[1]]))) / len(i[1])],
                'ratio_colon': [np.mean([int(str(x).endswith(':')) for x in i[1]])],
                'ratio_contain_number': [np.mean([int(any(char.isdigit() for char in str(x))) for x in i[1]])],
                'ratio_is_number': [np.mean([int(type(x) in ['float', 'int']) for x in i[1]])],
                'ratio_nonempty': [np.mean([int(len(str(x)) > 0) for x in i[1]])],
            }).T

            self.obj = concat([self.obj, features])

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

        return self.obj.T


def transform_for_baseline(raw_dataframe):
    """
    Transform an unprocessed web table dataset to feature space according to baseline

    Args:
        Dataframe with columns raw and label
    Returns:
        Dataframe with columns raw, label and feature space (107 columns)
    """
    with_features = DataFrame()
    for _, row in raw_dataframe.iterrows():
        try:
            with_features = with_features.append(_BaselineSample(row).transform(), ignore_index=True)
        except IndexError:  # FIXME: only tables with min shape 2x2 in dataset!
            print(row['path'])

    return with_features


class _ApproachSample(object):
    def __init__(self, obj, base_path):
        super().__init__()
        self.obj = obj
        self.base_path = base_path

    def _preprocess_html(self):
        pass

    def _create_image_from_html(self):
        pass

    def _trim_table(self):
        pass

    def transform(self):
        """
        Generate image re for a single web table according to our approach

        Args:
            None
        Returns:
            Dataframe with raw, label and feture vector for a single web column
        """
        self.new_html = self._preprocess_html()
        img_path = self._create_image_from_html()
        features = DataFrame({
            'img_path': img_path,
            'new_html': self.new_html
        }).T
        self.obj = concat([self.obj, features])

        return self.obj.T


def transform_for_approach(raw_dataframe, dataset_dir):
    """
    Transform an unprocessed web table dataset to feature space according to our approach

    Args:
        Dataframe with columns raw and label
    Returns:
        Dataframe with columns raw, label and imagepath
        Generates image representations of web table
    """
    os.makedirs(dataset_dir, exist_ok=True)
    img_base_path = os.path.join(dataset_dir, 'images')
    os.makedirs(img_base_path, exist_ok=True)

    with_img_path = DataFrame()
    for _, row in raw_dataframe.iterrows():
        with_img_path = with_img_path.append(_ApproachSample(row, img_base_path).transform(), ignore_index=True)

    return with_img_path
