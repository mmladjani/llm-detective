// Offline UI-controller checks: node --test tests/web_activity.test.cjs
// Run the real inline scripts with a small DOM/fetch boundary, not a model call.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function page(file) {
  const elements = new Map(), timers = new Map();
  let now = 1000, nextTimer = 0;
  const element = selector => {
    if (!elements.has(selector)) elements.set(selector, {
      hidden: true, disabled: false, textContent: '', style: {},
      classList: {toggle() {}},
      value: selector === '#mode' ? 'llm' : 'case-008',
      addEventListener() {},
    });
    return elements.get(selector);
  };
  const context = vm.createContext({
    document: {querySelector: element, querySelectorAll: () => []},
    // Initial config/builder/case requests stay pending. Individual tests may
    // supply an error response; no network requests leave this test process.
    fetch: () => new Promise(() => {}),
    Date: {now: () => now},
    setInterval: callback => { timers.set(++nextTimer, callback); return nextTimer; },
    clearInterval: id => timers.delete(id),
  });
  const html = fs.readFileSync(path.join(__dirname, '..', 'web', file), 'utf8');
  const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
  vm.runInContext(script, context, {filename: file});
  const run = source => vm.runInContext(source, context);
  return {element, timers, context, run,
    advance(ms) { now += ms; for (const tick of timers.values()) tick(); }};
}

for (const file of ['index.html', 'board.html']) {
  const main = file === 'index.html';
  const pending = main ? 'requestBusy' : 'busy';
  const automatic = main ? 'autoRunning' : 'running';
  const status = main ? '#runHelp' : '#workStatus';
  const investigating = `sid='test'; view={state:{status:'investigating'},run_metadata:{requested_mode:'llm'},awaiting_human:false};`;

  test(`${file}: spinner and elapsed clock follow a pending request`, () => {
    const p = page(file);
    p.run('syncControls()');
    assert.equal(p.element('#workSpinner').hidden, true);
    p.run(`${investigating} ${pending}=true; syncControls();`);
    assert.equal(p.element('#workSpinner').hidden, false);
    assert.equal(p.element('#start').disabled, true);
    assert.match(p.element(status).textContent, /Waiting for the AI/);
    p.advance(5000);
    assert.match(p.element('#workElapsed').textContent, /^5s elapsed/);
    p.run('syncControls(); syncControls();');
    assert.equal(p.timers.size, 1, 're-rendering must not create extra clocks');
    p.run(`${pending}=false; syncControls();`);
    assert.equal(p.element('#workSpinner').hidden, true);
    assert.equal(p.element('#workElapsed').hidden, true);
    assert.equal(p.timers.size, 0);
    assert.match(p.element(status).textContent, /paused/);
  });

  test(`${file}: auto-run stays active between turns, but not during a human pause`, () => {
    const p = page(file);
    p.run(`${investigating} ${automatic}=true; syncControls();`);
    assert.equal(p.element('#workSpinner').hidden, false);
    p.run(`view.state.status='awaiting_human'; view.awaiting_human=true; syncControls();`);
    assert.equal(p.element('#workSpinner').hidden, true);
    assert.equal(p.timers.size, 0);
    assert.match(p.element(status).textContent, main ? /Your turn/ : /witness answer/);
    p.run(`${pending}=true; ${main ? "requestKind='answer';" : ''} syncControls();`);
    assert.equal(p.element('#workSpinner').hidden, false, 'submitting an answer is a request');
  });

  test(`${file}: Pause must not hide a still-running request`, () => {
    const p = page(file);
    p.run(`${investigating} ${pending}=true; ${automatic}=false; syncControls();`);
    assert.equal(p.element('#workSpinner').hidden, false);
    p.run(`${pending}=false; syncControls();`);
    assert.equal(p.element('#workSpinner').hidden, true);
  });

  for (const outcome of ['solved', 'unresolved', 'conclusion_rejected', 'budget_exhausted', 'execution_error']) {
    test(`${file}: ${outcome} clears activity, even before the auto loop exits`, () => {
      const p = page(file);
      p.run(`${investigating} ${automatic}=true; syncControls();`);
      p.run(`view.state.status='${outcome}'; syncControls();`);
      assert.equal(p.element('#workSpinner').hidden, true);
      assert.equal(p.timers.size, 0);
      assert.match(p.element(status).textContent, /finished/);
    });
  }

  test(`${file}: a failed turn clears the spinner and reports the error`, async () => {
    const p = page(file);
    p.run(investigating);
    p.context.fetch = async () => { throw new Error('Simulated network failure'); };
    await p.run('step()');
    assert.equal(p.element('#workSpinner').hidden, true);
    assert.equal(p.timers.size, 0);
    assert.match(p.element(main ? '#err' : '#requestError').textContent, /Simulated network failure/);
  });
}
