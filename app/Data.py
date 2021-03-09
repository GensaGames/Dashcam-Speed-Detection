import os
import random

from app import Settings
import numpy as np

from app.other.LoggerFactory import get_logger


class Data:
    class Source:
        def __init__(self, name, path, y_values):
            self.name = name
            self.path = path
            self.amount = len(os.listdir(path))
            self.y_values = y_values

    class Holder:
        def __init__(self, train, validation):
            self.train = train
            self.validation = validation

    def __init__(self, batches=32):
        self.entries = []
        self.batches = batches

    def initialize(self, backward, train_part):
        get_logger().info(
            'Data is Initializing. Back: {} Part: {}'
                .format(backward, train_part))

        for source in SourceCombination.get_default():
            point = int(source.amount * train_part)

            train = np.arange(backward, point)
            np.random.shuffle(train)

            validation = np.arange(
                point + backward, source.amount)
            np.random.shuffle(validation)

            self.entries.append((
                source,
                Data.Holder(train, validation)
            ))
        return self

    def is_available(self):
        entries = list(filter(
            lambda x: len(x[1].train) >= self.batches,
            self.entries
        ))
        return len(entries) > 0

    def get_train_batch(self):
        try:
            batches = self.batches
            source, holder = random.choice(
                list(filter(
                    lambda x: len(x[1].train) >= batches,
                    self.entries
                ))
            )

            indexes = holder.train[:batches]
            holder.train = holder.train[batches:]
            return indexes, source, holder

        except IndexError:
            get_logger().info(
                'All entries were processed! Return None.')
            return None, None, None

    def get_validation_batch(self):
        batches = self.batches
        source, holder = random.choice(
            list(filter(
                lambda x: len(x[1].train) >= batches,
                self.entries
            ))
        )
        np.random.shuffle(holder.validation)
        indexes = holder.validation[:batches]
        return indexes, source, holder

    @staticmethod
    def summarize_y(y, frames_len):
        # Should Start from 20th element.
        value = [np.mean(a) for a in np.array_split(
            y[20:], frames_len)]
        assert len(value) == frames_len

        if frames_len < 2:
            return value

        # A bit of Magic
        for i in range(1, frames_len):
            f_delta = (value[i] - value[i - 1]) * (20/25)
            value[i] = value[i - 1] + f_delta

        return value


class SourceCombination:

    @staticmethod
    def get_all():
        array = [
            # 1. Source Train frames and Y values.
            Data.Source(
                'Default',
                Settings.RESOURCE + 'frames/',
                np.loadtxt(
                    Settings.RESOURCE + 'source/train.txt',
                    delimiter=" ")
            )
        ]
        # 2. Stop Frames and Zero-Y values
        location = Settings.RESOURCE + 'frames-stop/'
        for d1 in os.listdir(location):
            path = location + '/' + d1
            array.append(
                Data.Source(
                    'Stop-{}'.format(d1),
                    path,
                    np.zeros(len(os.listdir(path)))
                )
            )

        # 3. Custom CommaAi data
        location = Settings.CUSTOM + 'Chunk_1/'
        for i, d1 in enumerate(os.listdir(location)):
            for d2 in os.listdir(location + d1):
                outer = os.path.join(location, d1, d2)

                y = np.fromfile(os.path.join(
                    outer, 'processed_log/CAN/speed/value'
                ))
                path = os.path.join(outer, 'frames/')
                assert os.path.isdir(path)

                array.append(
                    Data.Source(
                        'Chunk1-{}-{}'.format(i, d2),
                        path,
                        Data.summarize_y(y, len(os.listdir(path)))
                    ),
                )

        return array

    @staticmethod
    def get_default():
        return [
            # 1. Source Train frames and Y values.
            Data.Source(
                'Default',
                Settings.RESOURCE + 'frames/',
                np.loadtxt(
                    Settings.RESOURCE + 'source/train.txt',
                    delimiter=" ")
            )]


if __name__ == "__main__":
    logger = get_logger()
    data = Data().initialize(10, 0.7)


    def print_info():
        logger.debug('Listing Data Sources!')
        for e, _ in data.entries:
            logger.debug('\n\n{}:'.format(e.name))
            logger.debug(
                '-> Path: {}. Y len: {}.'
                    .format(e.path, len(e.y_values))
            )
            logger.debug(
                '---> Y Max: {}. Y Min: {}.'
                    .format(max(e.y_values), min(e.y_values))
            )
        logger.debug('Listing done!\n\n')


    print_info()


    def validate():
        for _ in range(3):
            items, source, holder = data.get_train_batch()
            logger.debug('Validation {}.\nIndexes: {}'
                         .format(source.name, items[:5]))
            for i in items:
                assert i not in holder.train
                assert i not in holder.validation
        logger.debug('Validation Done successfully!')

    validate()
