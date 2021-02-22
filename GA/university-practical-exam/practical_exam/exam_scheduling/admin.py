from django.contrib import admin

from .models import *

# Register your models here.
admin.site.register(College)
admin.site.register(Department)
admin.site.register(Teacher)
admin.site.register(Student)
admin.site.register(Course)
admin.site.register(Lab)
admin.site.register(CourseOffering)
admin.site.register(CollegeHasDepartment)
admin.site.register(TimeTable)
admin.site.register(Allocation)
