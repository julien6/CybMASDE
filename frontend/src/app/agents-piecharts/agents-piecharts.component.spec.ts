import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AgentsPiechartsComponent } from './agents-piecharts.component';

describe('AgentsPiechartsComponent', () => {
  let component: AgentsPiechartsComponent;
  let fixture: ComponentFixture<AgentsPiechartsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ AgentsPiechartsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AgentsPiechartsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
