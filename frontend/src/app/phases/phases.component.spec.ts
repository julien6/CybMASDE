import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PhasesComponent } from './phases.component';

describe('PhasesComponent', () => {
  let component: PhasesComponent;
  let fixture: ComponentFixture<PhasesComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PhasesComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PhasesComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
