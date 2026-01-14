import datetime
from typing import Literal, List

import pandas as pd
from dateutil.relativedelta import relativedelta

from timeseries import date_to_utc
from timeseries.TimeseriesDate import TimeseriesDate


class TimeseriesInterval(TimeseriesDate):

    def __init__(self, date_from: datetime.datetime, date_to: datetime.datetime, time_unit: Literal['D', 'H', '15M']):
        date_from = date_to_utc(date_from)
        date_to = date_to_utc(date_to)
        super().__init__(date_from, time_unit)
        self.__date_to = date_to

    def get_date_from(self):
        return self.get_date()

    def get_date_to(self):
        return self.__date_to

    @staticmethod
    def create_from_steps(date: TimeseriesDate, steps: int = 0):
        if steps >= 0:
            return TimeseriesInterval(
                date_from=date.get_date(),
                date_to=date.get_next_timeseries_date(steps).get_date(),
                time_unit=date.time_unit
            )
        else:
            return TimeseriesInterval(
                date_from=date.get_next_timeseries_date(steps).get_date(),
                date_to=date.get_date(),
                time_unit=date.time_unit
            )

    def get_intervals(self, batch_size: int = 1):
        res: List[TimeseriesInterval] = []

        for i in range(int(self.get_steps_cnt() / batch_size) + 1):
            iter_date_from = self.get_next_timeseries_date(i * batch_size)
            iter_date_to = self.get_next_timeseries_date((i + 1) * batch_size - 1)

            if iter_date_to.get_date() > self.get_date_to():
                iter_date_to = TimeseriesDate(self.get_date_to(), self.time_unit)

            res.append(TimeseriesInterval(
                date_from=iter_date_from.get_date(),
                date_to=iter_date_to.get_date(),
                time_unit=self.time_unit
            ))

            if iter_date_to.get_date() == self.get_date_to():
                break

        return res

    def get_steps_cnt(self):
        if self.time_unit == 'D':
            time_range = pd.date_range(self.get_date_from(), self.get_date_to(), freq='d')
            cnt = len(time_range)
        elif self.time_unit == 'H':
            time_range = pd.date_range(self.get_date_from(), self.get_date_to(), freq='h')
            cnt = len(time_range)
        elif self.time_unit == '15M':
            time_range = pd.date_range(self.get_date_from(), self.get_date_to(), freq='m')
            cnt = int(len(time_range) / 15)
        else:
            raise Exception(f'Unknown time time_unit {self.time_unit}')

        return cnt

    def get_steps_range(self):
        return range(self.get_steps_cnt())

    def get_months_cnt(self):
        return relativedelta(self.get_date_to(), self.get_date_from()).months