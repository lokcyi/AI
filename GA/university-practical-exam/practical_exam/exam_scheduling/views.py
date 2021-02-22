import itertools
import json
import math

from django.db import IntegrityError, transaction
from django.db.models import Q
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods, require_safe

from . import utils
from .allocate import find_lab_allocations
from .forms import (
    CollegeDepartmentForm,
    CollegeDepartmentSemesterForm,
    DateTimeForm,
    DisplayForm,
    SchedulingForm,
    SeatingArrangementForm,
    TeacherForm,
)
from .models import (
    Allocation,
    College,
    CollegeHasDepartment,
    Course,
    CourseOffering,
    Department,
    Lab,
    Student,
    Teacher,
    TimeTable,
)
from .scheduling import get_res


@require_safe
def index(request):
    return render(request, "exam_scheduling/index.html")


@require_http_methods(["GET", "HEAD", "POST"])
@transaction.atomic
def schedule(request):
    if (
        request.method == "POST"
        and request.POST.get("scheduling_form", None) is not None
    ):
        scheduling_form = SchedulingForm(request.POST)

        if scheduling_form.is_valid():
            num_slots = scheduling_form.cleaned_data["num_slots"]

            sem = scheduling_form.cleaned_data["semesters"]
            semesters = semesters = Course.objects.distinct("semester").values_list(
                "semester", flat=True
            )

            if sem == "odd":
                semesters = [x for x in semesters if x % 2 == 1]
            elif sem == "even":
                semesters = [x for x in semesters if x % 2 == 0]

            courses_in_college = (
                CourseOffering.objects.filter(
                    teacher__college_dept_id=scheduling_form.cleaned_data[
                        "college_dept"
                    ]
                )
                .distinct("course__code")
                .values_list("course__code", flat=True)
            )
            courses = Course.objects.filter(
                semester__in=semesters, code__in=courses_in_college
            ).values("code", "semester")

            if len(courses) < num_slots:
                scheduling_form.add_error(
                    field=None, error="Insufficient number of courses to schedule"
                )
                return render(
                    request,
                    "exam_scheduling/schedule.html",
                    {"scheduling_form": scheduling_form},
                )

            results = get_res(1000, num_slots, courses)

            num_days = math.ceil(len(results) / num_slots)
            slot_nums = list(utils.chunks(results, num_slots))

            date_time_form = DateTimeForm(num_days=num_days, num_slots=num_slots)
            day_fields = [
                field for field in date_time_form if field.name.startswith("day")
            ]
            slot_fields = [
                field for field in date_time_form if field.name.startswith("slot")
            ]
            slot_fields = utils.chunks(slot_fields, 2)

            return render(
                request,
                "exam_scheduling/schedule.html",
                {
                    "scheduling_form": scheduling_form,
                    "college_dept": scheduling_form.cleaned_data["college_dept"].id,
                    "slots": slot_nums,
                    "num_days": num_days,
                    "num_slots": num_slots,
                    "date_time_form": date_time_form,
                    "day_fields": day_fields,
                    "slot_fields": slot_fields,
                },
            )

    elif (
        request.method == "POST"
        and request.POST.get("date_time_form", None) is not None
    ):
        date_time_form = DateTimeForm(request.POST)

        # college_dept = int(request.POST.get("college_dept"))
        college_dept = request.POST.get("college_dept")
        courses = json.loads(request.POST.get("slots").replace("'", '"'))

        try:
            with transaction.atomic():
                Allocation.objects.filter(
                    college_dept_day_slot__college_dept__exact=college_dept
                ).delete()
                TimeTable.objects.filter(college_dept__exact=college_dept).delete()

                for day_index, day in enumerate(courses):
                    for slot_index, slot in enumerate(day):
                        time_table = TimeTable(
                            college_dept=CollegeHasDepartment.objects.get(
                                id=college_dept
                            ),
                            day=day_index + 1,
                            slot=slot_index + 1,
                            course=Course.objects.get(code=slot),
                            date=date_time_form.cleaned_data[f"day{day_index + 1}"],
                            start_time=date_time_form.cleaned_data[
                                f"slot{slot_index + 1}_start_time"
                            ],
                            end_time=date_time_form.cleaned_data[
                                f"slot{slot_index + 1}_end_time"
                            ],
                        )

                        time_table.save()

        except IntegrityError as error:
            num_days = int(request.POST.get("num_days"))
            num_slots = int(request.POST.get("num_slots"))

            date_time_form.add_error(
                field=None,
                error="Failed to save timetable to database. "
                "Please enter proper date and time values.",
            )

            temp_form = DateTimeForm(num_days=num_days, num_slots=num_slots)
            day_fields = [field for field in temp_form if field.name.startswith("day")]
            slot_fields = [
                field for field in temp_form if field.name.startswith("slot")
            ]
            slot_fields = utils.chunks(slot_fields, 2)
            del temp_form

            return render(
                request,
                "exam_scheduling/schedule.html",
                {
                    "scheduling_form": SchedulingForm(),
                    "college_dept": college_dept,
                    "slots": courses,
                    "num_days": num_days,
                    "num_slots": num_slots,
                    "date_time_form": date_time_form,
                    "day_fields": day_fields,
                    "slot_fields": slot_fields,
                },
            )

        lab_allocations = find_lab_allocations(
            courses=list(itertools.chain.from_iterable(courses)),
            college_dept=college_dept,
        )
        for lab_allocation in lab_allocations:
            allocation = Allocation(
                college_dept_day_slot=TimeTable.objects.get(
                    college_dept=college_dept, course=lab_allocation["course"]
                ),
                lab=Lab.objects.get(
                    lab_id=lab_allocation["lab"], college_dept=college_dept
                ),
                int_examiner=Teacher.objects.get(id=lab_allocation["int_examiner"]),
                ext_examiner=Teacher.objects.get(id=lab_allocation["ext_examiner"]),
                start_usn=Student.objects.get(usn=lab_allocation["start_usn"]),
                end_usn=Student.objects.get(usn=lab_allocation["end_usn"]),
            )

            allocation.save()

        return redirect("exam_scheduling:display")

    else:
        return render(
            request,
            "exam_scheduling/schedule.html",
            {"scheduling_form": SchedulingForm()},
        )


@require_http_methods(["GET", "HEAD", "POST"])
def display(request):
    if request.method == "GET" or request.method == "HEAD":
        return render(
            request, "exam_scheduling/display.html", {"display_form": DisplayForm()}
        )

    # POST request
    if request.POST.get("display_form", None) is not None:
        display_form = DisplayForm(request.POST)

        if display_form.is_valid():
            if display_form.cleaned_data["choice"] == "dep":
                college_dept_form = CollegeDepartmentForm()
                return render(
                    request,
                    "exam_scheduling/display.html",
                    {
                        "display_form": display_form,
                        "college_dept_form": college_dept_form,
                    },
                )

            elif display_form.cleaned_data["choice"] == "sem":
                semester_form = CollegeDepartmentSemesterForm()
                return render(
                    request,
                    "exam_scheduling/display.html",
                    {"display_form": display_form, "semester_form": semester_form},
                )

            elif display_form.cleaned_data["choice"] == "teacher":
                teacher_form = TeacherForm()
                return render(
                    request,
                    "exam_scheduling/display.html",
                    {"display_form": display_form, "teacher_form": teacher_form},
                )

            elif display_form.cleaned_data["choice"] == "seating":
                seating_form = SeatingArrangementForm()
                return render(
                    request,
                    "exam_scheduling/display.html",
                    {"display_form": display_form, "seating_form": seating_form},
                )

    elif request.POST.get("college_dept_form", None) is not None:
        college_dept_form = CollegeDepartmentForm(request.POST)

        if college_dept_form.is_valid():
            college_dept = college_dept_form.cleaned_data["college_dept"]

            dates = (
                TimeTable.objects.filter(college_dept__exact=college_dept)
                .distinct("day")
                .order_by("day")
                .values_list("date", flat=True)
            )

            if not dates:
                college_dept_form.add_error(
                    field=None, error="No scheduled exams for current selection"
                )

                return render(
                    request,
                    "exam_scheduling/display.html",
                    {
                        "display_form": DisplayForm(),
                        "college_dept_form": college_dept_form,
                    },
                )

            times = (
                TimeTable.objects.filter(college_dept__exact=college_dept)
                .distinct("slot")
                .order_by("slot")
                .values("start_time", "end_time")
            )
            courses = (
                TimeTable.objects.filter(college_dept__exact=college_dept)
                .order_by("day", "slot")
                .values_list("course", flat=True)
            )
            courses = utils.chunks(courses, len(times))

            return render(
                request,
                "exam_scheduling/display.html",
                {
                    "display_form": DisplayForm(),
                    "college_dept_form": college_dept_form,
                    "dates": dates,
                    "times": times,
                    "courses": courses,
                },
            )

    elif request.POST.get("semester_form", None) is not None:
        semester_form = CollegeDepartmentSemesterForm(request.POST)

        if semester_form.is_valid():
            college_dept = semester_form.cleaned_data["college_dept"]
            dates = (
                TimeTable.objects.filter(college_dept__exact=college_dept)
                .distinct("day")
                .order_by("day")
                .values_list("date", flat=True)
            )
            times = (
                TimeTable.objects.filter(college_dept__exact=college_dept)
                .distinct("slot")
                .order_by("slot")
                .values("start_time", "end_time")
            )
            courses = (
                TimeTable.objects.filter(college_dept__exact=college_dept)
                .order_by("day", "slot")
                .values_list("course", flat=True)
            )

            sem_courses = Course.objects.filter(
                semester__exact=semester_form.cleaned_data["semester"]
            ).values_list("code", flat=True)

            if not any(course in sem_courses for course in courses):
                semester_form.add_error(
                    field=None, error="No scheduled exams for current selection"
                )

                return render(
                    request,
                    "exam_scheduling/display.html",
                    {"display_form": DisplayForm(), "semester_form": semester_form},
                )

            courses = [course if course in sem_courses else "" for course in courses]
            courses = utils.chunks(courses, len(times))

            return render(
                request,
                "exam_scheduling/display.html",
                {
                    "display_form": DisplayForm(),
                    "semester_form": semester_form,
                    "dates": dates,
                    "times": times,
                    "courses": courses,
                },
            )

    elif request.POST.get("teacher_form", None) is not None:
        teacher_form = TeacherForm(request.POST)

        if teacher_form.is_valid():
            teacher = teacher_form.cleaned_data["teacher"]

            college_dept_day_slots = list(
                Allocation.objects.filter(
                    Q(int_examiner__exact=teacher) | Q(ext_examiner__exact=teacher)
                )
                .distinct("college_dept_day_slot")
                .order_by("college_dept_day_slot")
                .values_list("college_dept_day_slot", flat=True)
            )

            if not college_dept_day_slots:
                teacher_form.add_error(
                    field=None, error="No scheduled exams for selected teacher"
                )

                return render(
                    request,
                    "exam_scheduling/display.html",
                    {"display_form": DisplayForm(), "teacher_form": teacher_form},
                )

            dates = (
                TimeTable.objects.filter(id__in=college_dept_day_slots)
                .order_by("day")
                .values_list("date", flat=True)
            )
            times = (
                TimeTable.objects.filter(id__in=college_dept_day_slots)
                .order_by("slot")
                .values("start_time", "end_time")
            )

            time_tables = TimeTable.objects.filter(
                id__in=college_dept_day_slots
            ).order_by("day", "slot")
            courses = [
                tt.course.code if tt.id in college_dept_day_slots else ""
                for tt in time_tables
            ]

            colleges = [
                tt.college_dept.college.code if tt.id in college_dept_day_slots else ""
                for tt in time_tables
            ]

            temp_labs = [
                allocation.lab.lab_id
                if allocation.college_dept_day_slot.id in college_dept_day_slots
                else ""
                for allocation in Allocation.objects.filter(
                    Q(int_examiner__exact=teacher) | Q(ext_examiner__exact=teacher)
                )
                .order_by("college_dept_day_slot")
                .distinct("college_dept_day_slot")
            ]

            labs = [""] * len(courses)
            lab_index = 0
            for course_index, course in enumerate(courses):
                if course:
                    labs[course_index] = temp_labs[lab_index]
                    lab_index += 1

            courses_colleges_labs = [
                {"course": course, "college": college, "lab": lab, "time": time}
                for course, college, lab, time in zip(courses, colleges, labs, times)
            ]
            # courses_colleges_labs = utils.chunks(courses_colleges_labs, len(times))

            return render(
                request,
                "exam_scheduling/display.html",
                {
                    "display_form": DisplayForm(),
                    "teacher_form": teacher_form,
                    "dates": dates,
                    "times": times,
                    "courses_colleges_labs": courses_colleges_labs,
                },
            )

    elif request.POST.get("seating_form", None) is not None:
        seating_form = SeatingArrangementForm(request.POST)

        if seating_form.is_valid():
            college_dept = seating_form.cleaned_data["college_dept"]
            date = seating_form.cleaned_data["date"]

            results = (
                Allocation.objects.filter(
                    college_dept_day_slot__college_dept__exact=college_dept,
                    college_dept_day_slot__date__exact=date,
                )
                .order_by("college_dept_day_slot", "lab")
                .values_list(
                    "college_dept_day_slot__course",
                    "lab__lab_id",
                    "start_usn__usn",
                    "end_usn__usn",
                )
            )

            if not results:
                seating_form.add_error(
                    field=None, error="No scheduled exams for current selection"
                )
                return render(
                    request,
                    "exam_scheduling/display.html",
                    {"display_form": DisplayForm(), "seating_form": seating_form},
                )

            courses_labs_usns = [
                {
                    "course": course,
                    "lab": lab,
                    "start_usn": start_usn,
                    "end_usn": end_usn,
                }
                for course, lab, start_usn, end_usn in results
            ]

            return render(
                request,
                "exam_scheduling/display.html",
                {
                    "display_form": DisplayForm(),
                    "seating_form": seating_form,
                    "courses_labs_usns": courses_labs_usns,
                },
            )
