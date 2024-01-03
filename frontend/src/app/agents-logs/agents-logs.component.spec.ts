import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AgentsLogsComponent } from './agents-logs.component';

describe('AgentsLogsComponent', () => {
  let component: AgentsLogsComponent;
  let fixture: ComponentFixture<AgentsLogsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ AgentsLogsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AgentsLogsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
