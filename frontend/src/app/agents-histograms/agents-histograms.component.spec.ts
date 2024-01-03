import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AgentsHistogramsComponent } from './agents-histograms.component';

describe('AgentsHistogramsComponent', () => {
  let component: AgentsHistogramsComponent;
  let fixture: ComponentFixture<AgentsHistogramsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ AgentsHistogramsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AgentsHistogramsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
