document.addEventListener('DOMContentLoaded', ()=>{
  const actions = {
    scan(){ alert('Scan Waste — placeholder action. Implement camera/AI here.'); },
    guide(){ alert('Segregation Guide — open guide page.'); },
    find(){ alert('Find Nearest Bin — open map or location finder.'); },
    report(){ alert('Report Issue — open report form.'); }
  };

  document.getElementById('scan').addEventListener('click', actions.scan);
  document.getElementById('guide').addEventListener('click', actions.guide);
  document.getElementById('find').addEventListener('click', actions.find);
  document.getElementById('report').addEventListener('click', actions.report);

  const logout = document.querySelector('.logout');
  logout.addEventListener('click', ()=>alert('Logged out (placeholder)'));
});