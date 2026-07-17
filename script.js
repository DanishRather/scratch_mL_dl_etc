const input = document.getElementById('todoInput');
const addBtn = document.getElementById('addBtn');
const todoList = document.getElementById('todoList');
const taskCount = document.getElementById('taskCount');
const clearBtn = document.getElementById('clearBtn');

let tasks = JSON.parse(localStorage.getItem('tasks')) || [];

function render() {
  todoList.innerHTML = '';
  if (tasks.length === 0) {
    todoList.innerHTML = '<li class="empty-msg">No tasks yet. Add one above!</li>';
  } else {
    tasks.forEach((task, index) => {
      const li = document.createElement('li');
      li.className = 'todo-item';
      li.innerHTML = `
        <div class="left">
          <input type="checkbox" ${task.done ? 'checked' : ''}>
          <span class="text ${task.done ? 'done' : ''}">${escapeHtml(task.text)}</span>
        </div>
        <button class="delete-btn">&times;</button>
      `;
      const checkbox = li.querySelector('input[type="checkbox"]');
      checkbox.addEventListener('change', () => toggleTask(index));
      li.querySelector('.delete-btn').addEventListener('click', () => deleteTask(index));
      todoList.appendChild(li);
    });
  }
  const done = tasks.filter(t => t.done).length;
  taskCount.textContent = `${tasks.length} tasks (${done} done)`;
  localStorage.setItem('tasks', JSON.stringify(tasks));
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function addTask() {
  const text = input.value.trim();
  if (!text) return;
  tasks.push({ text, done: false });
  input.value = '';
  input.focus();
  render();
}

function toggleTask(index) {
  tasks[index].done = !tasks[index].done;
  render();
}

function deleteTask(index) {
  tasks.splice(index, 1);
  render();
}

function clearAll() {
  if (tasks.length === 0) return;
  if (confirm('Delete all tasks?')) {
    tasks = [];
    render();
  }
}

addBtn.addEventListener('click', addTask);
input.addEventListener('keydown', e => { if (e.key === 'Enter') addTask(); });
clearBtn.addEventListener('click', clearAll);

render();
