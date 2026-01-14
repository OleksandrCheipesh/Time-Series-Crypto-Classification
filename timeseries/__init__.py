import datetime


def date_to_utc(date: datetime.datetime):
    return date.replace(tzinfo=datetime.timezone.utc)
