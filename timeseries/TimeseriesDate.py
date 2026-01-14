import datetime
from typing import Literal

from timeseries import date_to_utc


class TimeseriesDate:
    def __init__(self, date: datetime.datetime, time_unit: Literal['D', 'H', '15M']):
        self.__date = date_to_utc(date)
        self.time_unit: Literal['D', 'H', '15M'] = time_unit

    def get_date(self):
        return self.__date

    def get_next_timeseries_date(self, steps: int) -> 'TimeseriesDate':
        if steps == 0:
            return self

        if self.time_unit == 'D':
            d = self.__date + datetime.timedelta(days=steps)
        elif self.time_unit == 'H':
            d = self.__date + datetime.timedelta(hours=steps)
        elif self.time_unit == '15M':
            d = self.__date + datetime.timedelta(minutes=steps * 15)
        else:
            raise Exception(f'Unknown time time_unit {self.time_unit}')

        return TimeseriesDate(
            date=d,
            time_unit=self.time_unit
        )

    def get_time_unit_seconds(self):
        return 60 * 60 * 24 if self.time_unit == 'D' else 60 * 60 if self.time_unit == 'H' else 60 * 15
