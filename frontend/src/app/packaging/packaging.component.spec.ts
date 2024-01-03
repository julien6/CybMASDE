import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PackagingComponent } from './packaging.component';

describe('PackagingComponent', () => {
  let component: PackagingComponent;
  let fixture: ComponentFixture<PackagingComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ PackagingComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PackagingComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
