import datetime

import dateutil
from django import forms
from django.db.models import Max, Min
from django.http import QueryDict

from .models import College, CollegeHasDepartment, Course, Department, Teacher


class SchedulingForm(forms.Form):
    college_dept = forms.ModelChoiceField(
        queryset=CollegeHasDepartment.objects.all(),
        required=True,
        label="College and Department",
    )
    num_slots = forms.IntegerField(
        min_value=1, max_value=5, required=True, label="Number Of Slots"
    )

    # TODO: Calculate semester programmatically
    # TODO: Use enums if taking input via form
    semesters = forms.ChoiceField(
        choices=[("even", "Even"), ("odd", "Odd")], required=True, label="Semesters"
    )


class DateTimeForm(forms.Form):
    def __init__(self, *args, **kwargs):
        if "num_days" in kwargs and "num_slots" in kwargs:
            num_days = kwargs.pop("num_days")
            num_slots = kwargs.pop("num_slots")

            super().__init__(*args, **kwargs)

            today = datetime.datetime.today()
            now = datetime.datetime.combine(
                today, datetime.time(hour=datetime.datetime.now().hour)
            )

            for i in range(num_days):
                self.fields[f"day{i + 1}"] = forms.DateField(
                    initial=today,
                    required=True,
                    label=f"Day {i + 1}",
                )

            for i in range(num_slots):
                self.fields[f"slot{i + 1}_start_time"] = forms.TimeField(
                    initial=now,
                    required=True,
                    label=f"Slot {i + 1} start time",
                )
                self.fields[f"slot{i + 1}_end_time"] = forms.TimeField(
                    initial=now,
                    required=True,
                    label=f"Slot {i + 1} end time",
                )

        # Request data
        elif type(args[0]) is QueryDict:
            args = args[0]
            num_days = int(args.get("num_days"))
            num_slots = int(args.get("num_slots"))

            super().__init__(args, **kwargs)

            self.cleaned_data = {}

            for i in range(num_days):
                field_name = f"day{i + 1}"
                self.cleaned_data[field_name] = dateutil.parser.parse(
                    args.get(field_name)
                ).date()

            for i in range(num_slots):
                field_name = f"slot{i + 1}_start_time"
                self.cleaned_data[field_name] = dateutil.parser.parse(
                    args.get(field_name)
                ).time()

                field_name = f"slot{i + 1}_end_time"
                self.cleaned_data[field_name] = dateutil.parser.parse(
                    args.get(field_name)
                ).time()


class DisplayForm(forms.Form):
    choice = forms.ChoiceField(
        choices=[
            ("dep", "Department-wise"),
            ("sem", "Semester-wise"),
            ("teacher", "Teacher-wise"),
            ("seating", "Seating arrangement"),
        ],
        required=True,
    )


class CollegeDepartmentForm(forms.Form):
    college_dept = forms.ModelChoiceField(
        queryset=CollegeHasDepartment.objects.all(),
        required=True,
        label="College and Department",
    )


class CollegeDepartmentSemesterForm(forms.Form):
    college_dept = forms.ModelChoiceField(
        queryset=CollegeHasDepartment.objects.all(),
        required=True,
        label="College and Department",
    )
    semester = forms.IntegerField(
        max_value=Course.objects.aggregate(Max("semester"))["semester__max"],
        min_value=Course.objects.aggregate(Min("semester"))["semester__min"],
        required=True,
    )


class TeacherForm(forms.Form):
    teacher = forms.ModelChoiceField(queryset=Teacher.objects.all(), required=True)
    # TODO: Add college_dept to filter by college_dept


class SeatingArrangementForm(forms.Form):
    college_dept = forms.ModelChoiceField(
        queryset=CollegeHasDepartment.objects.all(),
        required=True,
        label="College and Department",
    )
    date = forms.DateField(
        initial=datetime.datetime.today(),
        required=True,
    )
