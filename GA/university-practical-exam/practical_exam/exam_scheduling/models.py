import pgtrigger
from django.db import models
from django.utils.translation import gettext_lazy as _


@pgtrigger.register(
    pgtrigger.Trigger(
        name="uppercase_college_code",
        operation=pgtrigger.Insert | pgtrigger.Update,
        when=pgtrigger.Before,
        func="NEW.code=upper(NEW.code); RETURN NEW;",
    ),
    pgtrigger.Protect(
        name="protect_college_deletes",
        operation=pgtrigger.Delete,
    ),
)
class College(models.Model):
    class Region(models.TextChoices):
        NORTH = "N", _("North")
        SOUTH = "S", _("South")
        EAST = "E", _("East")
        WEST = "W", _("West")

    code = models.CharField(max_length=10, primary_key=True)
    name = models.CharField(max_length=100)
    address = models.CharField(max_length=500)
    region = models.CharField(
        max_length=1, choices=Region.choices, default=Region.NORTH
    )

    def __repr__(self):
        return self.name

    def __str__(self):
        return repr(self)


@pgtrigger.register(
    pgtrigger.Trigger(
        name="uppercase_department_code",
        operation=pgtrigger.Insert | pgtrigger.Update,
        when=pgtrigger.Before,
        func="NEW.code=upper(NEW.code); RETURN NEW;",
    ),
    pgtrigger.Protect(
        name="protect_department_deletes",
        operation=pgtrigger.Delete,
    ),
)
class Department(models.Model):
    code = models.CharField(max_length=4, primary_key=True)
    name = models.CharField(max_length=20, unique=True)

    def __repr__(self):
        return self.name

    def __str__(self):
        return repr(self)


class CollegeHasDepartment(models.Model):
    college = models.ForeignKey(College, on_delete=models.CASCADE, to_field="code")
    department = models.ForeignKey(
        Department, on_delete=models.CASCADE, to_field="code"
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["college", "department"], name="college_dept"
            )
        ]

    def __repr__(self):
        return f"{self.college.name} - {self.department}"

    def __str__(self):
        return repr(self)


@pgtrigger.register(
    pgtrigger.Trigger(
        name="uppercase_teacher_id",
        operation=pgtrigger.Insert | pgtrigger.Update,
        when=pgtrigger.Before,
        func="NEW.id=upper(NEW.id); RETURN NEW;",
    )
)
class Teacher(models.Model):
    id = models.CharField(max_length=20, primary_key=True, verbose_name="ID")
    name = models.CharField(max_length=50)
    college_dept = models.ForeignKey(
        CollegeHasDepartment,
        on_delete=models.CASCADE,
        to_field="id",
        verbose_name="College and Department",
    )
    years_experience = models.PositiveSmallIntegerField(
        verbose_name="years of experience"
    )

    def __repr__(self):
        return f"{self.id} ({self.name})"

    def __str__(self):
        return repr(self)


@pgtrigger.register(
    pgtrigger.Trigger(
        name="uppercase_student_usn",
        operation=pgtrigger.Insert | pgtrigger.Update,
        when=pgtrigger.Before,
        func="NEW.usn=upper(NEW.usn); RETURN NEW;",
    )
)
class Student(models.Model):
    usn = models.CharField(max_length=20, primary_key=True)
    name = models.CharField(max_length=50)
    admission_year = models.PositiveSmallIntegerField(verbose_name="year of admission")
    college_dept = models.ForeignKey(
        CollegeHasDepartment,
        on_delete=models.CASCADE,
        to_field="id",
        verbose_name="College and Department",
    )
    semester = models.PositiveSmallIntegerField()

    def __repr__(self):
        return f"{self.usn} ({self.name})"

    def __str__(self):
        return repr(self)


@pgtrigger.register(
    pgtrigger.Trigger(
        name="uppercase_course_code",
        operation=pgtrigger.Insert | pgtrigger.Update,
        when=pgtrigger.Before,
        func="NEW.code=upper(NEW.code); RETURN NEW;",
    )
)
class Course(models.Model):
    class CourseType(models.TextChoices):
        CORE = "Core"
        ELECTIVE = "Elective"

    code = models.CharField(max_length=5, primary_key=True)
    name = models.CharField(max_length=100)
    department = models.ForeignKey(
        Department, on_delete=models.CASCADE, to_field="code"
    )
    semester = models.PositiveSmallIntegerField()
    course_type = models.CharField(
        max_length=8,
        verbose_name="Type",
        choices=CourseType.choices,
        default=CourseType.CORE,
    )
    num_credits = models.PositiveSmallIntegerField(verbose_name="Number of credits")

    def __repr__(self):
        return f"{self.code} ({self.name})"

    def __str__(self):
        return repr(self)


@pgtrigger.register(
    pgtrigger.Trigger(
        name="uppercase_lab_id",
        operation=pgtrigger.Insert | pgtrigger.Update,
        when=pgtrigger.Before,
        func="NEW.lab_id=upper(NEW.lab_id); RETURN NEW;",
    )
)
class Lab(models.Model):
    lab_id = models.CharField(max_length=10, verbose_name="ID")
    college_dept = models.ForeignKey(
        CollegeHasDepartment,
        on_delete=models.CASCADE,
        to_field="id",
        verbose_name="College and Department",
    )
    capacity = models.PositiveSmallIntegerField()

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["lab_id", "college_dept"], name="college_lab"
            )
        ]

    def __repr__(self):
        return f"{repr(self.college_dept)} - {self.lab_id}"

    def __str__(self):
        return repr(self)


class CourseOffering(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE, to_field="code")
    teacher = models.ForeignKey(Teacher, on_delete=models.CASCADE, to_field="id")

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["course", "teacher"], name="course_teacher")
        ]

    def __repr__(self):
        return f"{self.course} - {repr(self.teacher)}"

    def __str__(self):
        return repr(self)


class TimeTable(models.Model):
    college_dept = models.ForeignKey(
        CollegeHasDepartment,
        on_delete=models.CASCADE,
        to_field="id",
        verbose_name="College and Department",
    )
    day = models.PositiveSmallIntegerField()
    slot = models.PositiveSmallIntegerField()
    course = models.ForeignKey(Course, on_delete=models.CASCADE, to_field="code")
    date = models.DateField()
    start_time = models.TimeField()
    end_time = models.TimeField()

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["college_dept", "day", "slot"],
                name="college_dept_day_slot",
            ),
            models.UniqueConstraint(
                fields=["college_dept", "date", "start_time", "end_time"],
                name="college_dept_date_time",
            ),
            models.UniqueConstraint(
                fields=["college_dept", "course"],
                name="college_dept_course",
            ),
        ]

    def __repr__(self):
        return f"{repr(self.college_dept)} - {str(self.date)} - {repr(self.course)}"

    def __str__(self):
        return repr(self)


class Allocation(models.Model):
    college_dept_day_slot = models.ForeignKey(
        TimeTable,
        on_delete=models.CASCADE,
        to_field="id",
        verbose_name="College and Department",
    )
    lab = models.ForeignKey(Lab, on_delete=models.SET_NULL, to_field="id", null=True)
    int_examiner = models.ForeignKey(
        Teacher,
        on_delete=models.SET_NULL,
        to_field="id",
        null=True,
        verbose_name="Internal Examiner",
        related_name="int_examiner",
    )
    ext_examiner = models.ForeignKey(
        Teacher,
        on_delete=models.SET_NULL,
        to_field="id",
        null=True,
        verbose_name="External Examiner",
        related_name="ext_examiner",
    )
    start_usn = models.ForeignKey(
        Student,
        on_delete=models.SET_NULL,
        to_field="usn",
        null=True,
        related_name="start_usn",
    )
    end_usn = models.ForeignKey(
        Student,
        on_delete=models.SET_NULL,
        to_field="usn",
        null=True,
        related_name="end_usn",
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["college_dept_day_slot", "lab"],
                name="college_dept_day_slot_allocation",
            ),
            models.UniqueConstraint(
                fields=["college_dept_day_slot", "int_examiner"],
                name="college_dept_day_slot_int_examiner",
            ),
            models.UniqueConstraint(
                fields=["college_dept_day_slot", "ext_examiner"],
                name="college_dept_day_slot_ext_examiner",
            ),
        ]

    def __repr__(self):
        return f"{repr(self.college_dept_day_slot)} - {repr(self.lab)}"

    def __str__(self):
        return repr(self)
