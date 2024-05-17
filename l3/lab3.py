from l3.task3 import task3
from task1 import task1
from task2 import task2


with open('doc11.txt', 'r', encoding='utf-8') as file:
    documents = [line.strip() for line in file if line.strip()]

task1_result = task1(documents)
task2_result = task2(documents)
task3_result = task3(documents, task2_result)

print(f'MARINER vector: {task1_result}')
print(f'Corpuses: {task2_result}')
for x in task3_result:
    print(f'{x}: {task3_result[x]}')