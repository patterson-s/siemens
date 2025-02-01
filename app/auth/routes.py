from flask import render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from app.auth import bp
from app.models.models import User, LoginLog
from app.auth.forms import LoginForm
from urllib.parse import urlparse
from datetime import datetime
from app import db

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        
        # Log the login attempt
        log = LoginLog(
            user_id=user.id if user else None,
            attempted_username=form.username.data,
            success=False,
            ip_address=request.remote_addr,
            user_agent=request.user_agent.string
        )
        db.session.add(log)
        
        if user is None:
            db.session.commit()
            flash('Invalid username or password', 'danger')
            return redirect(url_for('auth.login'))
        
        # Check if user is locked
        if user.is_locked():
            remaining_time = user.locked_until - datetime.utcnow()
            minutes = int(remaining_time.total_seconds() / 60)
            flash(f'Account is locked. Please try again in {minutes} minutes.', 'danger')
            db.session.add(log)
            db.session.commit()
            return redirect(url_for('auth.login'))
            
        if not user.check_password(form.password.data):
            user.increment_failed_login()
            db.session.add(log)
            db.session.commit()
            
            if user.is_locked():
                flash('Too many failed attempts. Account is locked for 1 hour.', 'danger')
            else:
                remaining_attempts = 5 - user.failed_login_count
                flash(f'Invalid password. {remaining_attempts} attempts remaining.', 'danger')
            return redirect(url_for('auth.login'))
        
        # Successful login
        user.reset_failed_login()
        log.success = True
        db.session.add(log)
        db.session.commit()
        
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or urlparse(next_page).netloc != '':
            next_page = url_for('main.index')
        return redirect(next_page)
    
    return render_template('auth/login.html', title='Sign In', form=form)

@bp.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('main.index')) 