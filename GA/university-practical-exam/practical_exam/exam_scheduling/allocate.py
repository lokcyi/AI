from typing import List

from .models import (
    College,
    CollegeHasDepartment,
    Course,
    CourseOffering,
    Department,
    Lab,
    Student,
    Teacher,
)


def find_lab_allocations(courses: List[str], college_dept: int):
    labs = Lab.objects.filter(college_dept__id__exact=college_dept).values(
        "lab_id", "capacity"
    )

    semesters = (
        Course.objects.filter(code__in=courses)
        .distinct("semester")
        .values_list("semester", flat=True)
    )

    internal_teachers, external_teachers = get_teachers(
        min_experience=3,
        college_dept=college_dept,
        courses=courses,
    )

    internal_teachers, external_teachers = initialize_teachers(
        internal_teachers=internal_teachers, external_teachers=external_teachers
    )

    lab_allocations = []
    slot_num = 0
    for course in courses:
        students = (
            Student.objects.filter(
                college_dept__id__exact=college_dept,
                semester__exact=Course.objects.get(code=course).semester,
            )
            .order_by("usn")
            .values_list("usn", flat=True)
        )

        remaining_students = len(students)
        total_students = len(students)

        for lab in labs:
            slot_num += 1
            if lab["capacity"] < remaining_students:
                (
                    int_teacher,
                    ext_teacher,
                    internal_teachers,
                    external_teachers,
                ) = update_and_return_teachers(
                    internal_teachers=internal_teachers,
                    external_teachers=external_teachers,
                    course=course,
                    slot_num=slot_num,
                )

                lab_allocations.append(
                    {
                        "course": course,
                        "lab": lab["lab_id"],
                        "int_examiner": int_teacher,
                        "ext_examiner": ext_teacher,
                        "start_usn": students[total_students - remaining_students],
                        "end_usn": students[
                            total_students - remaining_students + lab["capacity"] - 1
                        ],
                    }
                )
                remaining_students -= lab["capacity"]

            else:
                (
                    int_teacher,
                    ext_teacher,
                    internal_teachers,
                    external_teachers,
                ) = update_and_return_teachers(
                    internal_teachers=internal_teachers,
                    external_teachers=external_teachers,
                    course=course,
                    slot_num=slot_num,
                )
                lab_allocations.append(
                    {
                        "course": course,
                        "lab": lab["lab_id"],
                        "int_examiner": int_teacher,
                        "ext_examiner": ext_teacher,
                        "start_usn": students[total_students - remaining_students],
                        "end_usn": students[total_students - 1],
                    }
                )
                break

    return lab_allocations


def get_teachers(min_experience: int, courses: List[str], college_dept: int):
    internal_teachers = (
        CourseOffering.objects.filter(
            course__code__in=courses,
            teacher__id__in=Teacher.objects.filter(
                college_dept__id__exact=college_dept,
                years_experience__gte=min_experience,
            ).values_list("id", flat=True),
        )
        .distinct("teacher__id")
        .values_list("teacher__id", flat=True)
    )

    region = CollegeHasDepartment.objects.get(id=college_dept).college.region
    department = CollegeHasDepartment.objects.get(id=college_dept).department.code

    external_teachers = (
        CourseOffering.objects.filter(
            course__code__in=courses,
            teacher__id__in=Teacher.objects.filter(
                college_dept__college__region__exact=region,
                college_dept__department__code__exact=department,
                years_experience__gte=min_experience,
            )
            .exclude(college_dept__id__exact=college_dept)
            .values_list("id", flat=True),
        )
        .distinct("teacher__id")
        .values_list("teacher__id", flat=True)
    )

    return (internal_teachers, external_teachers)


def initialize_teachers(internal_teachers: List[str], external_teachers: List[str]):
    internal_teachers = {i: 0 for i in internal_teachers}
    external_teachers = {i: 0 for i in external_teachers}

    return (internal_teachers, external_teachers)


def update_and_return_teachers(
    internal_teachers: dict, external_teachers: dict, course: str, slot_num: int
):
    selected_int_teacher = None
    selected_ext_teacher = None

    for teacher in internal_teachers.keys():
        if CourseOffering.objects.filter(
            course__code=course, teacher__id=teacher
        ).exists():
            selected_int_teacher = teacher
            internal_teachers[teacher] = slot_num
            internal_teachers = dict(
                sorted(internal_teachers.items(), key=lambda item: item[1])
            )
            break

    for teacher in external_teachers.keys():
        if CourseOffering.objects.filter(
            course__code=course, teacher__id=teacher
        ).exists():
            selected_ext_teacher = teacher
            external_teachers[teacher] = slot_num
            external_teachers = dict(
                sorted(external_teachers.items(), key=lambda item: item[1])
            )
            break

    return (
        selected_int_teacher,
        selected_ext_teacher,
        internal_teachers,
        external_teachers,
    )
